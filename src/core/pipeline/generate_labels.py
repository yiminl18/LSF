#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Labels Module

Generates provenance labels for the dataset using Embeddings + LLM Judge.
Reads processing/embeddings from SHARED.
Writes labels to SHARED.

Usage:
    python -m core.pipeline.generate_labels --dataset pdfs --limit 30 --workers 4 --parser docling
"""

import argparse
import json
import sys
import os
import queue
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.utils.progress import create_pipeline_progress_with_cost
from core.retrieval.retrieval import find_provenance_node
from core.retrieval.judge_header import JUDGE_MODES, ContentFilterError
from core.llm.cost import get_prices
from core.llm.model import LLM_PROVIDERS

EMBED_PROVIDERS = [
    "openai",
    "azure",
    "openrouter",
]


@dataclass
class LabelEntry:
    """Single label entry."""

    doc_name: str
    question_idx: int
    question: str
    ground_truth: str
    possible_provenance_nodes: List[Dict[str, Any]] = field(default_factory=list)
    max_checked_rank: int = 0
    # SciBench extension fields (backward-compatible: defaults used when missing from old label files)
    absence_label: bool = False  # True = this information does not exist in the document (answer="No" etc.)
    question_category: str = ""  # Question category, e.g., "Title", "Ethical Statement Included?"
    answer_type: str = ""  # Answer format, e.g., "Text", "Yes/No", "List of names"


def load_questions(path: Path) -> List[str]:
    """Load questions list."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def match_pdf_to_ground_truth(
    pdf_dir: Path,
    gt_dir: Path,
) -> List[Tuple[Path, str, List[str]]]:
    """Match PDF files with GT answers."""
    matched = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc_name = pdf_path.stem
        gt_filename = f"{doc_name}.txt_answers.json"
        gt_path = gt_dir / gt_filename

        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                answers = json.load(f)
            # Sort by numeric key; supports any number of answers
            max_key = max((int(k) for k in answers if k.isdigit()), default=0)
            answer_list = []
            for i in range(1, max_key + 1):
                ans = answers.get(str(i), "")
                if isinstance(ans, list):
                    ans = ", ".join(str(item) for item in ans)
                answer_list.append(str(ans))
            matched.append((pdf_path, doc_name, answer_list))

    return matched


def load_existing_labels(
    path: Path,
    force_without_absence: bool = False,
) -> Tuple[List[LabelEntry], Set[Tuple[str, int]]]:
    """Load existing labels cache.

    When force_without_absence=True, only absence entries are skipped;
    entries with existing provenance or checked status will be reprocessed.
    """
    if not path.exists():
        return [], set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = []
        processed = set()
        valid_fields = {f.name for f in fields(LabelEntry)}
        for item in data.get("labels", []):
            filtered = {k: v for k, v in item.items() if k in valid_fields}
            label = LabelEntry(**filtered)
            labels.append(label)
            if force_without_absence:
                # Only skip absence entries; reprocess everything else
                is_done = label.absence_label
            else:
                # Absence entries, entries with provenance, or already-checked entries count as "processed"
                is_done = (
                    label.absence_label
                    or label.possible_provenance_nodes
                    or label.max_checked_rank > 0
                )
            if is_done:
                processed.add((label.doc_name, label.question_idx))

        return labels, processed
    except Exception as e:
        logging.warning(f"Failed to load label cache {path}: {e}")
        return [], set()


def save_labels(labels: List[LabelEntry], path: Path) -> None:
    """Save labels to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": len(labels),
        "labels": [asdict(label) for label in labels],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_labels_filename(
    q_idx: int, label_tag: Optional[str], dataset: str = "pdfs"
) -> str:
    """Build label filename. Format: {prefix}_q{idx}[_{tag}]_reconstructed_labels.json"""
    from core.config import get_label_prefix

    prefix = get_label_prefix(dataset)
    parts = [f"{prefix}_q{q_idx}"]
    if label_tag:
        parts.append(label_tag)
    parts.append("reconstructed")
    return "_".join(parts) + "_labels.json"


def _estimate_cost(
    dataset: str,
    q_idxs_to_run: List[int],
    doc_limit: Optional[int],
    top_k: int,
    llm_provider: str,
    llm_model: str,
    label_tag: Optional[str],
    experiment: str,
    judge_mode: str = "answer_compare",
    parser: str = "docling",
    force_without_absence: bool = False,
) -> Tuple[float, int]:
    """
    Estimate LLM judge cost (excluding embedding cost).

    answer_compare: each header -> ask() + 70% equal_llm(), up to 2 calls
    support_judge:  each header -> 1 direct judgment call, ~50-60% cheaper
    """
    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)
    pdf_dir = paths.get_data_dir(dataset)
    gt_dir = paths.get_ground_truth_dir(dataset)
    labels_dir = paths.get_labels_dir(dataset)
    questions = load_questions(paths.get_questions_path(dataset))

    price_in, price_out = get_prices(llm_provider, model=llm_model)

    matched_docs = match_pdf_to_ground_truth(pdf_dir, gt_dir)
    if doc_limit is not None:
        matched_docs = matched_docs[:doc_limit]

    total_headers = 0

    for q_idx in q_idxs_to_run:
        if q_idx >= len(questions):
            continue

        # Check existing cache
        labels_path = labels_dir / build_labels_filename(
            q_idx, label_tag, dataset=dataset
        )
        _, processed = load_existing_labels(
            labels_path, force_without_absence=force_without_absence
        )

        for _pdf_path, doc_name, answers in matched_docs:
            if (doc_name, q_idx) in processed:
                continue

            gt = answers[q_idx] if q_idx < len(answers) else ""
            if not gt or gt.lower() == "none":
                continue

            # Read processing JSON to count headers
            processing_path = paths.get_processing_json_path(dataset, doc_name)
            if not processing_path.exists():
                continue

            try:
                with open(processing_path, "r", encoding="utf-8") as f:
                    struct_data = json.load(f)
                headers = [
                    t
                    for t in struct_data.get("texts", [])
                    if t.get("label") == "section_header"
                ]
                n_headers = min(len(headers), top_k)
                total_headers += n_headers
            except Exception:
                continue

    # Cost estimation
    if judge_mode == "support_judge":
        # support_judge: 1 call per header, ~300 tokens input / ~5 tokens output
        total_input = total_headers * 300
        total_output = total_headers * 5
        total_calls = total_headers
    else:
        # answer_compare: ask() ~500 in/~50 out + 70% equal_llm() ~200 in/~5 out
        eq_calls = int(total_headers * 0.7)
        total_input = total_headers * 500 + eq_calls * 200
        total_output = total_headers * 50 + eq_calls * 5
        total_calls = total_headers + eq_calls

    cost = total_input * (price_in / 1_000_000) + total_output * (price_out / 1_000_000)

    return cost, total_calls


def _generate_labels_worker(args):
    """Worker function for parallel processing."""
    (
        dataset,
        q_idx,
        doc_limit,
        experiment,
        top_k,
        progress_queue,
        label_tag,
        judge_mode,
        llm_provider,
        llm_model,
        match_limit,
        embed_provider,
        page,
        parser,
        force_without_absence,
    ) = args

    try:
        variant = parser if parser != "docling" else None
        paths = PathManager(experiment=experiment, processing_variant=variant)

        # Set up log file
        log_dir = paths.get_logs_dir(dataset)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"q{q_idx}.log"

        logger = logging.getLogger(f"q{q_idx}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(fh)

        # Reset cost counters
        from core.embed.embeddings import reset_embedding_cost, get_embedding_cost
        from core.llm.model import reset_llm_cost, get_llm_cost

        reset_embedding_cost()
        reset_llm_cost(llm_provider)

        logger.info(
            f"Question {q_idx} started - top_k={top_k}, match_limit={match_limit}, embed={embed_provider}, doc_limit={doc_limit}, llm={llm_provider}, model={llm_model}"
        )

        # Read from SHARED
        pdf_dir = paths.get_data_dir(dataset)
        gt_dir = paths.get_ground_truth_dir(dataset)
        questions_path = paths.get_questions_path(dataset)

        # Read processing/embeddings from SHARED (select directory by embed_provider)
        embeddings_dir = paths.get_embeddings_dir(dataset, provider=embed_provider)

        # Write labels to SHARED
        labels_dir = paths.get_labels_dir(dataset)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Load Question
        questions = load_questions(questions_path)
        if q_idx >= len(questions):
            if progress_queue:
                progress_queue.put((q_idx, "log", f"[red]Q{q_idx} Out of bounds[/red]"))
            return []
        question = questions[q_idx]

        # Match Docs
        matched_docs = match_pdf_to_ground_truth(pdf_dir, gt_dir)
        if doc_limit is not None:
            matched_docs = matched_docs[:doc_limit]

        # Labels File Path
        labels_path = labels_dir / build_labels_filename(
            q_idx, label_tag, dataset=dataset
        )

        # Load Cache
        labels, processed = load_existing_labels(
            labels_path, force_without_absence=force_without_absence
        )
        if processed:
            labels = [l for l in labels if l.question_idx == q_idx]
            processed = {(d, q) for d, q in processed if q == q_idx}
            if progress_queue:
                progress_queue.put((q_idx, "log", f"Cached: {len(labels)}"))

        if progress_queue:
            progress_queue.put((q_idx, "progress", 0, len(matched_docs)))

        new_count = 0
        for i, (pdf_path, doc_name, answers) in enumerate(matched_docs):
            if progress_queue:
                progress_queue.put((q_idx, "progress", i, len(matched_docs)))

            if (doc_name, q_idx) in processed:
                logger.info(f"[SKIP] {doc_name} - already processed")
                continue

            ground_truth = answers[q_idx] if q_idx < len(answers) else ""

            if not ground_truth or ground_truth.lower() == "none":
                logger.info(f"[SKIP] {doc_name} - no ground truth")
                continue

            # Processing JSON Path
            processing_path = paths.get_processing_json_path(dataset, doc_name)
            if not processing_path.exists():
                logger.info(f"[SKIP] {doc_name} - missing processing file")
                if progress_queue:
                    progress_queue.put(
                        (
                            q_idx,
                            "log",
                            f"[yellow]Missing Processing {doc_name}[/yellow]",
                        )
                    )
                continue

            logger.info(f"Processing {doc_name}...")

            # Find Provenance
            try:
                matches, checked_count = find_provenance_node(
                    pdf_path=str(pdf_path),
                    merged_json_path=str(processing_path),
                    question=question,
                    answer=ground_truth,
                    cache_dir=str(embeddings_dir),
                    top_k_check=top_k,
                    judge_mode=judge_mode,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    match_limit=match_limit,
                    provider=embed_provider,
                    header_page=page,
                )

                provenance_nodes = []
                if matches:
                    match_text = matches[0]["header"].get("text", "")[:50]
                    rank = matches[0]["rank_of_accepted"]
                    escaped_match = match_text.replace('"', '\\"')
                    logger.info(f'[MATCH] Found at rank {rank}: "{escaped_match}"')
                    if progress_queue:
                        progress_queue.put(
                            (
                                q_idx,
                                "log",
                                f"[green]Found {doc_name}[/green]: {match_text[:30]}",
                            )
                        )

                    for m in matches:
                        header = m["header"]
                        path_text = (header.get("structure") or {}).get(
                            "path_text", ""
                        ) or header.get("path_text", "")
                        node_info = {
                            "header_idx": m["header_idx"],
                            "text": header.get("text", ""),
                            "text_span": header.get("text_span", ""),
                            "path_text": path_text,
                            "rank": m["rank_of_accepted"],
                            "similarity": m["similarity"],
                            "gap_from_top1": m["gap_from_top1"],
                            "margin_to_next": m["margin_to_next"],
                        }
                        provenance_nodes.append(node_info)
                else:
                    logger.info(f"[MISS] No provenance found (checked {checked_count})")
                    if progress_queue:
                        progress_queue.put(
                            (q_idx, "log", f"[yellow]Miss {doc_name}[/yellow]")
                        )

                # Find existing placeholder entry and update it (preserving SciBench metadata)
                existing_idx = next(
                    (
                        i
                        for i, l in enumerate(labels)
                        if l.doc_name == doc_name and l.question_idx == q_idx
                    ),
                    None,
                )
                if existing_idx is not None:
                    labels[existing_idx].possible_provenance_nodes = provenance_nodes
                    labels[existing_idx].max_checked_rank = checked_count
                else:
                    label = LabelEntry(
                        doc_name=doc_name,
                        question_idx=q_idx,
                        question=question,
                        ground_truth=ground_truth,
                        possible_provenance_nodes=provenance_nodes,
                        max_checked_rank=checked_count,
                    )
                    labels.append(label)
                processed.add((doc_name, q_idx))
                new_count += 1

                # Incremental Save
                save_labels(labels, labels_path)

                # Send cost update
                emb_tokens, emb_cost = get_embedding_cost()
                llm_cost = get_llm_cost(llm_provider)
                if progress_queue:
                    progress_queue.put((q_idx, "cost", emb_tokens, emb_cost, llm_cost))

            except ContentFilterError as e:
                # Content filter error: skip this document and continue with others
                logger.warning(f"[CONTENT_FILTER] {doc_name}: {e}")
                if progress_queue:
                    progress_queue.put(
                        (
                            q_idx,
                            "log",
                            f"[yellow]CONTENT_FILTER {doc_name}: skipped[/yellow]",
                        )
                    )
                continue
            except Exception as e:
                # Other errors: raise immediately to stop processing
                logger.error(f"[ERROR] {doc_name}: {e}")
                if progress_queue:
                    progress_queue.put(
                        (q_idx, "log", f"[red]ERROR {doc_name}: {e}[/red]")
                    )
                raise  # Re-raise to let the main process know to stop

        # Final cost summary
        emb_tokens, emb_cost = get_embedding_cost()
        llm_cost = get_llm_cost(llm_provider)
        total_cost = emb_cost + llm_cost
        logger.info(
            f"Completed - Embedding: {emb_tokens} tokens (${emb_cost:.4f}), LLM: ${llm_cost:.4f}, Total: ${total_cost:.4f}"
        )

        if progress_queue:
            progress_queue.put(
                (q_idx, "progress", len(matched_docs), len(matched_docs))
            )
            progress_queue.put((q_idx, "cost", emb_tokens, emb_cost, llm_cost))

        return labels

    except Exception as e:
        # Any error is re-raised to let the main process know to stop
        if progress_queue:
            progress_queue.put((q_idx, "log", f"[red]CRITICAL WORKER ERROR: {e}[/red]"))
        raise  # Re-raise to stop the entire pipeline


def generate_labels(
    dataset: str = "pdfs",
    limit: Optional[int] = 30,
    doc_limit: Optional[int] = None,
    experiment: str = "default",
    top_k: int = 50,
    workers: Optional[int] = None,
    query_idx: Optional[list[int]] = None,
    label_tag: Optional[str] = None,
    judge_mode: str = "answer_compare",
    llm_provider: str = "azure",  # CLI layer requires this; Python API keeps a default for run_pipeline.py compatibility
    llm_model: Optional[str] = None,
    match_limit: int = 3,
    embed_provider: str = "openrouter",
    page: Optional[int] = None,
    yes: bool = False,
    parser: str = "docling",
    force_without_absence: bool = False,
) -> dict:
    """Generate labels for questions."""
    if not llm_model:
        raise ValueError("llm_model must be specified explicitly")

    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)
    questions_path = paths.get_questions_path(dataset)
    questions = load_questions(questions_path)

    # Determine questions to run
    if query_idx is not None:
        q_idxs_to_run = query_idx
    else:
        q_idxs_to_run = list(range(min(limit, len(questions))))

    if workers is None:
        workers = min(os.cpu_count() or 1, len(q_idxs_to_run))
    if judge_mode not in JUDGE_MODES:
        raise ValueError(
            f"Unsupported judge_mode={judge_mode}. Expected one of {sorted(JUDGE_MODES)}"
        )

    print("=== Generate Labels [Problem 1 Step 4] ===")
    print(f"Dataset:    {dataset}")
    print(f"LLM:        {llm_provider}")
    print(f"Model:      {llm_model}")
    print(f"Match Lim:  {match_limit}")
    print(f"Page:       {page if page is not None else 'All'}")
    print(f"Embed:      {embed_provider}")
    print(f"Questions:  {len(q_idxs_to_run)} (Workers: {workers})")
    print(f"Docs/Q:     {doc_limit if doc_limit else 'All'}")
    print(f"Label Tag:  {label_tag or '(default)'}")
    print(f"Judge Mode: {judge_mode}")
    print(f"Output:     {paths.get_labels_dir(dataset)} (SHARED)")
    print()

    # Cost estimation
    estimated_cost, estimated_calls = _estimate_cost(
        dataset,
        q_idxs_to_run,
        doc_limit,
        top_k,
        llm_provider,
        llm_model,
        label_tag,
        experiment,
        judge_mode,
        parser,
        force_without_absence,
    )
    print("=== Cost Estimation ===")
    print(f"Estimated judge calls: {estimated_calls:,}")
    print(f"Estimated cost:        ${estimated_cost:.4f}")
    if not yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            return {}

    results = {}

    # Cost tracking
    worker_costs = {}

    with Manager() as manager:
        progress_queue = manager.Queue()

        tasks = []
        for q_idx in q_idxs_to_run:
            tasks.append(
                (
                    dataset,
                    q_idx,
                    doc_limit,
                    experiment,
                    top_k,
                    progress_queue,
                    label_tag,
                    judge_mode,
                    llm_provider,
                    llm_model,
                    match_limit,
                    embed_provider,
                    page,
                    parser,
                    force_without_absence,
                )
            )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_q_idx = {
                executor.submit(_generate_labels_worker, task): task[1]
                for task in tasks
            }

            def cost_getter() -> float:
                return sum(c[1] + c[2] for c in worker_costs.values())

            with create_pipeline_progress_with_cost(cost_getter) as progress:
                main_task_id = progress.add_task(
                    "[bold green]Total Questions", total=len(tasks)
                )
                question_task_ids = {}

                completed_count = 0

                while completed_count < len(tasks):
                    try:
                        while not progress_queue.empty():
                            msg = progress_queue.get_nowait()
                            q_idx, type_ = msg[0], msg[1]

                            if type_ == "progress":
                                current, total = msg[2], msg[3]
                                if q_idx not in question_task_ids:
                                    question_task_ids[q_idx] = progress.add_task(
                                        f"Q{q_idx}", total=total
                                    )
                                progress.update(
                                    question_task_ids[q_idx], completed=current
                                )

                            elif type_ == "log":
                                message = msg[2]
                                progress.console.print(f"[Q{q_idx}] {message}")

                            elif type_ == "cost":
                                emb_tokens, emb_cost, llm_cost = msg[2], msg[3], msg[4]
                                worker_costs[q_idx] = (emb_tokens, emb_cost, llm_cost)

                    except queue.Empty:
                        pass

                    done_futures = [f for f in future_to_q_idx if f.done()]
                    for f in done_futures:
                        q_idx = future_to_q_idx.pop(f)
                        completed_count += 1
                        progress.advance(main_task_id)

                        if q_idx in question_task_ids:
                            progress.update(question_task_ids[q_idx], visible=False)

                        try:
                            res = f.result()
                            results[q_idx] = len(res)
                        except Exception as e:
                            # Any error stops the entire pipeline
                            progress.console.print(
                                f"[red]CRITICAL: Error in Q{q_idx}: {e}[/red]"
                            )
                            progress.console.print("[red]Stopping all workers...[/red]")
                            # Cancel all pending tasks
                            for future in future_to_q_idx:
                                if not future.done():
                                    future.cancel()
                            raise  # re-raise to stop main function

                    import time

                    time.sleep(0.1)

    # Final cost summary
    if worker_costs:
        total_emb_tokens = sum(c[0] for c in worker_costs.values())
        total_emb_cost = sum(c[1] for c in worker_costs.values())
        total_llm_cost = sum(c[2] for c in worker_costs.values())
        total_cost = total_emb_cost + total_llm_cost
        print()
        print("=== Cost Summary ===")
        print(f"Embedding:  {total_emb_tokens:,} tokens (${total_emb_cost:.4f})")
        print(f"LLM Judge:  ${total_llm_cost:.4f} ({llm_provider})")
        print(f"Total:      ${total_cost:.4f}")
        print(f"Logs:       {paths.get_logs_dir(dataset)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Labels")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--llm-provider",
        type=str,
        required=True,
        choices=sorted(LLM_PROVIDERS),
        help="LLM provider for judge",
    )
    parser.add_argument("--model", type=str, required=True, help="LLM model for judge")
    parser.add_argument("--limit", type=int, default=30, help="Num questions")
    parser.add_argument("--doc-limit", type=int, help="Num docs per question")
    parser.add_argument("--query-idx", type=str, help="Query indices, e.g. 8 or 8,9")
    parser.add_argument(
        "--experiment", type=str, default="default", help="Experiment ID"
    )
    parser.add_argument("--workers", type=int, help="Workers")
    parser.add_argument("--top-k", type=int, default=50, help="Top K retrieval")
    parser.add_argument(
        "--match-limit", type=int, default=3, help="Max matches per question"
    )
    parser.add_argument(
        "--page",
        type=int,
        help="Only retrieve section_headers on the specified page (page numbers match the structural JSON)",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="openrouter",
        choices=EMBED_PROVIDERS,
        help="Embedding provider",
    )
    parser.add_argument("--label-tag", type=str, help="Label file suffix tag")
    parser.add_argument(
        "--judge-mode",
        type=str,
        required=True,
        choices=sorted(JUDGE_MODES),
        help="Provenance judge strategy (answer_compare: two-step QA + semantic comparison; support_judge: single direct judgment)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip interactive confirmation and proceed automatically"
    )
    parser.add_argument(
        "--force-without-absence",
        action="store_true",
        help="Force reprocessing of non-absence entries (ignore existing provenance / checked status)",
    )
    parser.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        dest="struct_parser",
        help="Structural parser (docling/mineru)",
    )

    args = parser.parse_args()

    generate_labels(
        dataset=args.dataset,
        limit=args.limit,
        doc_limit=args.doc_limit,
        experiment=args.experiment,
        workers=args.workers,
        top_k=args.top_k,
        query_idx=[int(x) for x in args.query_idx.split(",")]
        if args.query_idx
        else None,
        label_tag=args.label_tag,
        judge_mode=args.judge_mode,
        llm_provider=args.llm_provider,
        llm_model=args.model,
        match_limit=args.match_limit,
        embed_provider=args.embed_provider,
        page=args.page,
        yes=args.yes,
        parser=args.struct_parser,
        force_without_absence=args.force_without_absence,
    )


if __name__ == "__main__":
    main()
