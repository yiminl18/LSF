#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Embeddings Module

Generates vector embeddings for all documents in the dataset.
Reads processing JSON (reconstructed.json) from SHARED datasets directory.
Writes embeddings to SHARED datasets directory.

Usage:
    python -m core.pipeline.generate_embeddings --dataset pdfs --parser docling
"""

import argparse
import sys
import queue
import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

import tiktoken

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.utils.progress import (
    create_pipeline_progress,
    create_pipeline_progress_with_cost,
    managed_progress,
)
from core.utils.io import suppress_stdout_stderr
from core.config import EMBEDDING_PRICE_PER_MILLION
from core.embed.embeddings import (
    build_embedding,
    get_embedding_cost,
    get_combined_text,
    reset_embedding_cost,
    resolve_embedding_batch_size,
    is_all_zero_vector,
    _load_embeddings_npz,
    _save_embeddings_npz,
)

# Supported embedding providers
EMBED_PROVIDERS = [
    "openai",
    "azure",
    "openrouter",
]


def _clean_all_zero_embeddings_from_npz(npz_path: Path) -> tuple[int, int]:
    """
    Clean all-zero embeddings from a single npz file.

    Returns:
        (total_vectors_before, removed_count)
    """
    if not npz_path.exists():
        return 0, 0

    try:
        embeddings = _load_embeddings_npz(npz_path)
        total_before = len(embeddings)

        keys_to_remove = []
        for key, vec in embeddings.items():
            if is_all_zero_vector(vec):
                keys_to_remove.append(key)

        if keys_to_remove:
            for key in keys_to_remove:
                del embeddings[key]
            _save_embeddings_npz(npz_path, embeddings)

        return total_before, len(keys_to_remove)
    except Exception:
        return 0, 0


def _pre_clean_all_zero_embeddings(cache_dir: Path) -> tuple[int, int]:
    """
    Pre-clean all-zero embeddings from all npz files in cache directory.

    Returns:
        (total_removed, total_vectors_scanned)
    """
    npz_files = list(cache_dir.glob("*_embeddings.npz"))
    if not npz_files:
        return 0, 0

    total_removed = 0
    total_vectors = 0

    print("=== Pre-cleaning All-Zero Embeddings ===")
    print(f"Scanning {len(npz_files)} embedding files...")

    for npz_path in npz_files:
        before, removed = _clean_all_zero_embeddings_from_npz(npz_path)
        total_vectors += before
        total_removed += removed
        if removed > 0:
            print(f"  {npz_path.name}: removed {removed} all-zero vectors")

    if total_removed > 0:
        print(
            f"\nTotal: {total_removed} all-zero vectors removed from {total_vectors} total vectors.\n"
        )
    else:
        print(f"No all-zero vectors found in {total_vectors} total vectors.\n")

    return total_removed, total_vectors


def _extract_embedding_texts(processing_path: Path) -> list[str]:
    """Extract deduplicated texts from a single document that need embedding."""
    with open(processing_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    texts = merged_data.get("texts", [])
    from core.config import MAX_SPAN_WORDS

    headers = [
        t
        for t in texts
        if isinstance(t, dict)
        and (t.get("label") in (None, "section_header"))
        and len(t.get("text_span", "").split()) < MAX_SPAN_WORDS
    ]
    if not headers:
        return []

    current_texts: list[str] = []
    seen: set[str] = set()
    for header in headers:
        combined_text = get_combined_text(header)
        if combined_text and combined_text not in seen:
            current_texts.append(combined_text)
            seen.add(combined_text)
        path_text = (header.get("structure") or {}).get("path_text", "") or header.get(
            "path_text", ""
        )
        if path_text and path_text not in seen:
            current_texts.append(path_text)
            seen.add(path_text)
    return current_texts


def _load_cached_keys(npz_path: Path) -> set[str]:
    """Load the set of text keys already cached in the npz file (for cost estimation deduplication)."""
    if not npz_path.exists():
        return set()

    import numpy as np

    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "keys" in data.files:
                return set(str(k) for k in data["keys"])
            return set()
    except Exception:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "keys" in data.files:
                    return set(str(k) for k in data["keys"])
                return set()
        except Exception:
            return set()


def _estimate_openai_embedding_cost(
    processing_files: list[Path],
    cache_dir: Path,
    force: bool,
) -> tuple[int, int, int, float]:
    """Estimate total tokens and cost for OpenAI embedding."""
    try:
        encoder = tiktoken.encoding_for_model("text-embedding-3-small")
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    total_docs = 0
    total_texts = 0
    total_tokens = 0

    for processing_path in processing_files:
        stem = processing_path.stem
        npz_path = cache_dir / f"{stem}_embeddings.npz"
        cached_keys = set() if force else _load_cached_keys(npz_path)

        current_texts = _extract_embedding_texts(processing_path)
        to_add_texts = [t for t in current_texts if t not in cached_keys]
        if not to_add_texts:
            continue

        total_docs += 1
        total_texts += len(to_add_texts)
        total_tokens += sum(len(encoder.encode(t)) for t in to_add_texts)

    estimated_cost = total_tokens * (EMBEDDING_PRICE_PER_MILLION / 1_000_000)
    return total_docs, total_texts, total_tokens, estimated_cost


def _build_worker(args):
    """Worker for parallel processing"""
    (
        processing_path,
        cache_dir,
        provider,
        use_sliding_window,
        force,
        batch_size,
        progress_queue,
    ) = args
    stem = Path(processing_path).stem
    doc_name = stem
    if doc_name.endswith("_reconstructed"):
        doc_name = doc_name[: -len("_reconstructed")]
    # Use npz cache only
    npz_path = Path(cache_dir) / f"{stem}_embeddings.npz"
    legacy_json_path = Path(cache_dir) / f"{stem}_embeddings.json"

    try:
        tokens_before, cost_before = get_embedding_cost()
        # Clean up legacy json cache to avoid false cache-hit detection
        legacy_json_path.unlink(missing_ok=True)
        if force:
            npz_path.unlink(missing_ok=True)

        # Suppress internal tqdm
        with suppress_stdout_stderr():
            total_emb, skipped = build_embedding(
                str(processing_path),
                str(cache_dir),
                provider=provider,
                use_sliding_window=use_sliding_window,
                show_progress=False,
                batch_size=batch_size,
            )

        tokens_after, cost_after = get_embedding_cost()
        delta_tokens = max(0, tokens_after - tokens_before)
        delta_cost = max(0.0, cost_after - cost_before)
        if progress_queue:
            progress_queue.put(
                ("processed", doc_name, delta_tokens, delta_cost, skipped)
            )
        return "processed"
    except Exception as e:
        tokens_after, cost_after = get_embedding_cost()
        delta_tokens = max(0, tokens_after - tokens_before)
        delta_cost = max(0.0, cost_after - cost_before)
        if progress_queue:
            progress_queue.put(("failed", doc_name, str(e), delta_tokens, delta_cost))
        return f"failed: {e}"


def generate_embeddings(
    dataset: str = "pdfs",
    limit: Optional[int] = None,
    force: bool = False,
    experiment: str = "default",
    provider: str = "openrouter",
    use_sliding_window: bool = False,
    include_no_gt: bool = False,
    workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    yes: bool = False,
    parser: str = "docling",
) -> int:
    """
    Generate embeddings for all reconstructed JSONs.

    Args:
        dataset: Dataset name.
        limit: Max docs to process.
        force: Force rebuild.
        experiment: Experiment ID (unused for output).
        provider: Embedding provider.
        include_no_gt: Include documents without ground truth.
        batch_size: Embedding API batch size (uses provider default if not specified).
        workers: Parallel workers (API).
    """
    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)

    # Read processing JSON from SHARED
    source_dir = paths.get_processing_dir(dataset)
    # Write embeddings to SHARED
    cache_dir = paths.get_embeddings_dir(dataset, provider=provider)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pattern = "*_reconstructed.json"
    processing_files = sorted(source_dir.glob(pattern))

    gt_dir = paths.get_ground_truth_dir(dataset)
    if include_no_gt:
        valid_files = processing_files
    else:
        # Filter by GT availability
        valid_files = []
        for mf in processing_files:
            doc_name = mf.stem
            if doc_name.endswith("_reconstructed"):
                doc_name = doc_name[: -len("_reconstructed")]
            gt_path = gt_dir / f"{doc_name}.txt_answers.json"
            if gt_path.exists():
                valid_files.append(mf)

    if limit is not None:
        processing_files = valid_files[:limit]
    else:
        processing_files = valid_files

    # Determine execution mode
    if workers is None:
        workers = 1
    resolved_batch_size = resolve_embedding_batch_size(provider, batch_size)

    pre_cleaned_count, pre_cleaned_total = _pre_clean_all_zero_embeddings(cache_dir)

    print("=== Generate Embeddings [Problem 1 Step 3] ===")
    print(f"Dataset:    {dataset}")
    print(f"Provider:   {provider}")
    print(f"Input Dir:  {source_dir}")
    print(f"Output Dir: {cache_dir} (SHARED)")
    print(f"To Process: {len(processing_files)} docs")
    print(f"GT Filter:  {'OFF' if include_no_gt else 'ON'}")
    print(f"Workers:    {workers}")
    print(f"Batch Size: {resolved_batch_size}")
    print()

    estimated_tokens = 0
    estimated_cost = 0.0
    estimated_docs, estimated_texts, estimated_tokens, estimated_cost = (
        _estimate_openai_embedding_cost(
            processing_files=processing_files,
            cache_dir=cache_dir,
            force=force,
        )
    )
    print("=== Cost Estimation ===")
    print(f"Estimated docs:        {estimated_docs:,}")
    print(f"Estimated new vectors: {estimated_texts:,}")
    print(f"Estimated tokens:      {estimated_tokens:,}")
    print(f"Estimated cost:        ${estimated_cost:.4f}")
    if not yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            return 0
    reset_embedding_cost()
    print()

    processed = 0
    skipped = 0
    failed = 0
    mp_actual_tokens = 0
    mp_actual_cost = 0.0
    total_skipped_all_zero = 0

    if workers > 1:
        with Manager() as manager:
            progress_queue = manager.Queue()

            tasks = []
            for mf in processing_files:
                tasks.append(
                    (
                        str(mf),
                        str(cache_dir),
                        provider,
                        use_sliding_window,
                        force,
                        resolved_batch_size,
                        progress_queue,
                    )
                )

            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all
                for task in tasks:
                    executor.submit(_build_worker, task)

                # Rich Progress (managed_progress lets inner tqdm be reused)
                with managed_progress(create_pipeline_progress()) as progress:
                    task_id = progress.add_task(
                        "[green]Total Progress", total=len(tasks)
                    )

                    completed_count = 0
                    while completed_count < len(tasks):
                        try:
                            while not progress_queue.empty():
                                msg = progress_queue.get_nowait()
                                status = msg[0]

                                if status == "processed":
                                    processed += 1
                                    if len(msg) >= 4:
                                        mp_actual_tokens += int(msg[2])
                                        mp_actual_cost += float(msg[3])
                                    if len(msg) >= 5:
                                        total_skipped_all_zero += int(msg[4])
                                    progress.console.print(
                                        f"[green]✓ Processed[/green] {msg[1]}"
                                    )
                                    progress.advance(task_id)
                                    completed_count += 1
                                elif status == "skipped":
                                    skipped += 1
                                    progress.advance(task_id)
                                    completed_count += 1
                                elif status == "failed":
                                    failed += 1
                                    if len(msg) >= 5:
                                        mp_actual_tokens += int(msg[3])
                                        mp_actual_cost += float(msg[4])
                                    progress.console.print(
                                        f"[red]✗ Failed[/red] {msg[1]}: {msg[2]}"
                                    )
                                    progress.advance(task_id)
                                    completed_count += 1
                        except queue.Empty:
                            pass

                        import time

                        time.sleep(0.1)

    else:
        # Sequential (Rich) - managed_progress lets inner tqdm be reused
        # Cost-aware progress bar
        def cost_getter() -> float:
            _, cost = get_embedding_cost()
            return cost

        progress_bar = create_pipeline_progress_with_cost(cost_getter=cost_getter)

        with managed_progress(progress_bar) as progress:
            task_id = progress.add_task(
                "[green]Processing", total=len(processing_files)
            )

            for processing_path in processing_files:
                stem = processing_path.stem
                doc_name = stem
                if doc_name.endswith("_reconstructed"):
                    doc_name = doc_name[: -len("_reconstructed")]
                npz_cache = Path(cache_dir) / f"{stem}_embeddings.npz"
                legacy_json_cache = Path(cache_dir) / f"{stem}_embeddings.json"

                # Clean up legacy json cache to avoid false cache-hit detection
                legacy_json_cache.unlink(missing_ok=True)

                if force:
                    progress.console.print(
                        f"[yellow]⚠ Force Rebuild:[/yellow] {doc_name}"
                    )
                    npz_cache.unlink(missing_ok=True)

                try:
                    # Show inner embedding progress in single-worker mode to avoid long periods without visible feedback
                    show_inner = workers == 1
                    total_emb, skipped_all_zero = build_embedding(
                        str(processing_path),
                        str(cache_dir),
                        provider=provider,
                        use_sliding_window=use_sliding_window,
                        show_progress=show_inner,
                        batch_size=resolved_batch_size,
                    )
                    total_skipped_all_zero += skipped_all_zero
                    processed += 1
                    progress.console.print(f"[green]✓ Processed[/green] {doc_name}")
                except Exception as e:
                    progress.console.print(f"[red]✗ Failed[/red] {doc_name}: {e}")
                    failed += 1

                progress.advance(task_id)

    print("\n=== Embeddings Complete ===")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    if pre_cleaned_count > 0 or total_skipped_all_zero > 0:
        regenerated_success = pre_cleaned_count - total_skipped_all_zero
        print(f"\n{'=' * 80}")
        print("⚠️  Embedding Status Summary:")
        print(f"   - Pre-cleaned:    {pre_cleaned_count} all-zero vectors removed")
        print(
            f"   - Regenerated:     {regenerated_success} vectors regenerated successfully"
        )
        if total_skipped_all_zero > 0:
            print(
                f"   - Failed:         {total_skipped_all_zero} vectors still all-zero after {3} retries"
            )
        print(f"{'=' * 80}\n")

    if workers > 1:
        actual_tokens, actual_cost = mp_actual_tokens, mp_actual_cost
    else:
        actual_tokens, actual_cost = get_embedding_cost()
    print()
    print("=== Cost Summary ===")
    print(f"Estimated tokens:    {estimated_tokens:,}")
    print(f"Estimated cost:      ${estimated_cost:.4f}")
    print(f"Actual tokens:       {actual_tokens:,}")
    print(f"Actual cost:         ${actual_cost:.4f}")
    print(f"Delta cost:          ${(actual_cost - estimated_cost):.4f}")

    return processed + skipped


def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, help="Max docs")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment ID (unused for output)",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="openrouter",
        choices=EMBED_PROVIDERS,
        help=f"Embedding provider (choices: {', '.join(EMBED_PROVIDERS)})",
    )
    parser.add_argument(
        "--use-sliding-window",
        action="store_true",
        help="Use sliding window (unsupported for API providers)",
    )
    parser.add_argument(
        "--include-no-gt",
        action="store_true",
        help="Include documents without ground truth",
    )
    parser.add_argument("--workers", type=int, help="Workers (API)")
    parser.add_argument("--batch-size", type=int, help="Embedding batch size")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip interactive confirmation and proceed automatically"
    )
    parser.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        help="Structural parser (docling/mineru)",
    )

    args = parser.parse_args()

    generate_embeddings(
        dataset=args.dataset,
        limit=args.limit,
        force=args.force,
        experiment=args.experiment,
        provider=args.embed_provider,
        use_sliding_window=args.use_sliding_window,
        include_no_gt=args.include_no_gt,
        workers=args.workers,
        batch_size=args.batch_size,
        yes=args.yes,
        parser=args.parser,
    )


if __name__ == "__main__":
    main()
