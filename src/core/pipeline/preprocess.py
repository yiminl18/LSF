#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Preprocessing Module

Converts PDF files into intermediate JSON format.

Usage:
    python -m core.pipeline.preprocess --dataset pdfs --limit 10
"""

import argparse
import sys
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from core.utils.progress import managed_progress

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))


PARSER_DOCLING = "docling"
PARSER_MINERU = "mineru"
VALID_PARSERS = (PARSER_DOCLING, PARSER_MINERU)

# Structural parser output suffix mapping
_STRUCTURAL_SUFFIX = {
    PARSER_DOCLING: "_docling.json",
    PARSER_MINERU: "_mineru.json",
}


@lru_cache(maxsize=1)
def _get_converters(
    parser: str = PARSER_DOCLING,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Lazily load heavy parsing dependencies to avoid cold-start overhead
    when all documents are skipped.
    parser: structural parser selection (docling / mineru)
    """
    if parser == PARSER_MINERU:
        from core.doc.mineru_tool import to_json as structural_to_json
    else:
        from core.doc.docling_tool import to_json as structural_to_json
    from core.doc.lsf_tool import to_json as lsf_to_json

    return structural_to_json, lsf_to_json


def _configure_parser_logs(verbose_parser_logs: bool) -> None:
    """
    Configure parser log levels to prevent third-party logs from flooding
    the progress bar.
    """
    level = logging.INFO if verbose_parser_logs else logging.WARNING
    logging.getLogger("docling").setLevel(level)
    logging.getLogger("magic_pdf").setLevel(level)
    logging.getLogger("mineru").setLevel(level)


def _configure_hf_offline(hf_offline: bool) -> None:
    """
    Configure HuggingFace offline mode to avoid hitting the remote
    revision API on every initialization.
    """
    if not hf_offline:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _create_preprocess_progress() -> Progress:
    """
    Custom progress bar for preprocessing: prioritizes completed/total display
    with percentage shown to one decimal place.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    )


def _load_existing_processed_stems(
    output_dir: Path,
    parser: str = PARSER_DOCLING,
    run_type: Optional[str] = None,
) -> set[str]:
    """
    Scan output_dir and return the set of completed document stems.
    When run_type=None, both structural and lsf outputs must exist;
    when a specific type is given, only that file is checked.
    """
    lsf_suffix = "_lsf.json"
    if run_type == "lsf":
        return {p.name[: -len(lsf_suffix)] for p in output_dir.glob(f"*{lsf_suffix}")}
    if run_type in ("mineru", "docling"):
        suffix = _STRUCTURAL_SUFFIX[run_type]
        return {p.name[: -len(suffix)] for p in output_dir.glob(f"*{suffix}")}
    # run_type=None: skip only when both exist
    structural_suffix = _STRUCTURAL_SUFFIX[parser]
    structural_stems = {
        p.name[: -len(structural_suffix)]
        for p in output_dir.glob(f"*{structural_suffix}")
    }
    lsf_stems = {p.name[: -len(lsf_suffix)] for p in output_dir.glob(f"*{lsf_suffix}")}
    return structural_stems & lsf_stems


def _split_existing_and_pending(
    pdf_files: List[Path],
    output_dir: Path,
    force: bool,
    parser: str = PARSER_DOCLING,
    run_type: Optional[str] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Pre-check output files, separating already-existing (skippable)
    documents from those still pending.
    """
    if force:
        return [], pdf_files

    # Determine the display suffix after skipping based on run_type
    if run_type == "lsf":
        check_suffix = "_lsf.json"
    elif run_type in ("mineru", "docling"):
        check_suffix = _STRUCTURAL_SUFFIX[run_type]
    else:
        check_suffix = _STRUCTURAL_SUFFIX[parser]

    existing_stems = _load_existing_processed_stems(output_dir, parser, run_type)
    skipped_paths: List[Path] = []
    pending_pdf_files: List[Path] = []
    for pdf_path in pdf_files:
        if pdf_path.stem in existing_stems:
            skipped_paths.append(output_dir / f"{pdf_path.stem}{check_suffix}")
        else:
            pending_pdf_files.append(pdf_path)

    return skipped_paths, pending_pdf_files


def _filter_pdf_files_by_gt(
    pdf_files: List[Path], gt_dir: Path, include_no_gt: bool
) -> List[Path]:
    """
    Filter documents by ground truth availability; returns the original list
    unchanged when include_no_gt=True.
    """
    if include_no_gt:
        return pdf_files

    gt_suffix = ".txt_answers.json"
    gt_stems = {p.name[: -len(gt_suffix)] for p in gt_dir.glob(f"*{gt_suffix}")}
    return [pdf_path for pdf_path in pdf_files if pdf_path.stem in gt_stems]


def _ensure_parser_available(parser: str) -> None:
    """
    Validate structural parser dependencies before batch processing starts;
    provides a clear installation hint when missing.
    """
    if parser != PARSER_MINERU:
        return

    from core.doc.mineru_tool import get_mineru_install_hint, is_mineru_available

    if is_mineru_available():
        return

    raise RuntimeError(f"MinerU parser is not available in the current environment.\n{get_mineru_install_hint()}")


def _process_single_pdf(
    pdf_path_str: str,
    output_dir_str: str,
    force: bool,
    verbose_parser_logs: bool = False,
    hf_offline: bool = False,
    parser: str = PARSER_DOCLING,
    run_type: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Worker function for processing a single PDF (used for parallel execution).
    run_type: None=all, "lsf"=LSF only, "mineru"/"docling"=structural parsing only
    Returns: (status, result_path_str, error_msg)
    """
    try:
        pdf_path = Path(pdf_path_str)
        output_dir = Path(output_dir_str)
        doc_name = pdf_path.stem

        # Determine output file and skip logic based on run_type
        if run_type == "lsf":
            out_path = output_dir / f"{doc_name}_lsf.json"
        elif run_type in ("mineru", "docling"):
            out_path = output_dir / f"{doc_name}{_STRUCTURAL_SUFFIX[run_type]}"
        else:
            structural_suffix = _STRUCTURAL_SUFFIX[parser]
            structural_path = output_dir / f"{doc_name}{structural_suffix}"
            lsf_path = output_dir / f"{doc_name}_lsf.json"
            if structural_path.exists() and lsf_path.exists() and not force:
                return "skipped", str(structural_path), None
            out_path = structural_path

        if run_type is not None and out_path.exists() and not force:
            return "skipped", str(out_path), None

        _configure_hf_offline(hf_offline)
        _configure_parser_logs(verbose_parser_logs)

        if run_type == "lsf":
            from core.doc.lsf_tool import to_json as lsf_to_json

            lsf_to_json(str(pdf_path), output_dir=str(output_dir), verbose=False)
        elif run_type in ("mineru", "docling"):
            structural_to_json, _ = _get_converters(run_type)
            structural_to_json(str(pdf_path), output_dir=str(output_dir))
        else:
            structural_to_json, lsf_to_json = _get_converters(parser)
            structural_to_json(str(pdf_path), output_dir=str(output_dir))
            lsf_to_json(str(pdf_path), output_dir=str(output_dir), verbose=False)

        return "success", str(out_path), None

    except Exception as e:
        return "failed", None, str(e)


def preprocess_documents(
    dataset: str = "pdfs",
    limit: Optional[int] = None,
    force: bool = False,
    include_no_gt: bool = False,
    experiment: str = "00",
    workers: Optional[int] = None,
    verbose_parser_logs: bool = False,
    hf_offline: bool = False,
    run_type: Optional[str] = None,
) -> List[Path]:
    """
    Batch-process PDF files [Problem 1 Step 1].

    Args:
        dataset: Dataset name (pdfs, paper).
        limit: Maximum number of files to process; None processes all.
        force: Force reprocessing of already-existing files.
        include_no_gt: Whether to include documents without ground truth.
        experiment: Experiment name.
        workers: Number of parallel worker processes (None uses CPU count).
        verbose_parser_logs: Whether to show parser internal INFO logs.
        hf_offline: Whether to enable HuggingFace offline mode.
        run_type: Run type (None=all, "lsf"=LSF only, "mineru"/"docling"=structural parsing only).

    Returns:
        List of generated JSON paths.
    """
    # Determine the structural parser (default behavior when run_type=None)
    if run_type in ("mineru", "docling"):
        parser = run_type
    else:
        parser = PARSER_DOCLING

    if run_type != "lsf":
        _ensure_parser_available(parser)

    paths = PathManager(experiment=experiment)

    pdf_dir = paths.get_data_dir(dataset)
    output_dir = paths.get_processing_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    gt_dir = paths.get_ground_truth_dir(dataset)
    valid_pdf_files = _filter_pdf_files_by_gt(pdf_files, gt_dir, include_no_gt)

    if limit is not None:
        pdf_files = valid_pdf_files[:limit]
    else:
        pdf_files = valid_pdf_files

    if workers is None:
        workers = os.cpu_count() or 1
    _configure_hf_offline(hf_offline)

    type_label = run_type or f"all (structural={parser} + lsf)"
    print("=== PDF Preprocessing [Problem 1 Step 1] ===")
    print(f"Source:     {dataset}")
    print(f"Run Type:   {type_label}")
    print(f"PDF Dir:    {pdf_dir}")
    print(f"GT Dir:     {gt_dir} ({'GT filter OFF' if include_no_gt else 'GT filter ON'})")
    print(f"Output Dir: {output_dir}")
    if include_no_gt:
        print(
            f"To Scan:    {len(pdf_files)} files (selected from {len(valid_pdf_files)} PDFs)"
        )
    else:
        print(
            f"To Scan:    {len(pdf_files)} files (selected from {len(valid_pdf_files)} docs with ground truth)"
        )
    print(f"Workers:    {workers}")
    print(f"Parser Log: {'INFO' if verbose_parser_logs else 'WARNING'}")
    print(f"HF Mode:    {'OFFLINE' if hf_offline else 'ONLINE'}")
    print()

    results = []
    skipped_paths, pending_pdf_files = _split_existing_and_pending(
        pdf_files,
        output_dir,
        force,
        parser,
        run_type,
    )
    skipped = len(skipped_paths)
    results.extend(skipped_paths)
    failed = 0

    print(
        f"Pre-skip:   {skipped} (already exist)"
        if not force
        else "Pre-skip:   0 (--force enabled)"
    )
    print(f"To Process: {len(pending_pdf_files)}")

    if not pending_pdf_files:
        print("\n=== Preprocessing Complete ===")
        print(f"Success: {len(results) - skipped}")
        print(f"Skipped: {skipped} (already exist)")
        print(f"Failed:  {failed}")
        return results

    if workers <= 1:
        with managed_progress(_create_preprocess_progress()) as progress:
            task_id = progress.add_task("Processing PDFs", total=len(pending_pdf_files))
            for pdf_path in pending_pdf_files:
                doc_name = pdf_path.stem
                try:
                    progress.update(task_id, description=f"Processing PDF ({doc_name})")
                    status, path_str, error_msg = _process_single_pdf(
                        str(pdf_path),
                        str(output_dir),
                        force,
                        verbose_parser_logs,
                        hf_offline,
                        parser,
                        run_type,
                    )
                    if status == "success":
                        if path_str:
                            results.append(Path(path_str))
                    elif status == "skipped":
                        if path_str:
                            results.append(Path(path_str))
                        skipped += 1
                    elif status == "failed":
                        print(f"\n  ✗ {doc_name}: {error_msg}")
                        failed += 1
                except Exception as e:
                    print(f"\n  ✗ {doc_name}: System error - {e}")
                    failed += 1
                finally:
                    progress.advance(task_id)
                    progress.refresh()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Only submit documents that actually need processing to avoid scheduling overhead from mass skips
            future_to_pdf = {
                executor.submit(
                    _process_single_pdf,
                    str(p),
                    str(output_dir),
                    force,
                    verbose_parser_logs,
                    hf_offline,
                    parser,
                    run_type,
                ): p
                for p in pending_pdf_files
            }

            with managed_progress(_create_preprocess_progress()) as progress:
                task_id = progress.add_task("Processing PDFs", total=len(pending_pdf_files))
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    doc_name = pdf_path.stem

                    try:
                        progress.update(task_id, description=f"Processing PDF ({doc_name})")
                        status, path_str, error_msg = future.result()

                        if status == "success":
                            if path_str:
                                results.append(Path(path_str))
                        elif status == "skipped":
                            if path_str:
                                results.append(Path(path_str))
                            skipped += 1
                        elif status == "failed":
                            print(f"\n  ✗ {doc_name}: {error_msg}")
                            failed += 1
                    except Exception as e:
                        print(f"\n  ✗ {doc_name}: System error - {e}")
                        failed += 1
                    finally:
                        progress.advance(task_id)
                        progress.refresh()

    print("\n=== Preprocessing Complete ===")
    print(f"Success: {len(results) - skipped}")
    print(f"Skipped: {skipped} (already exist)")
    print(f"Failed:  {failed}")

    return results


def main():
    ap = argparse.ArgumentParser(description="PDF Preprocessing")
    ap.add_argument("--dataset", type=str, default="pdfs", help="Dataset name")
    ap.add_argument("--limit", type=int, help="Limit number of documents")
    ap.add_argument("--force", action="store_true", help="Force reprocessing")
    ap.add_argument("--include-no-gt", action="store_true", help="Include documents without GT")
    ap.add_argument("--experiment", type=str, default="00", help="Experiment name")
    ap.add_argument("--workers", type=int, help="Number of parallel workers (default: CPU count)")
    ap.add_argument("--verbose-parser-logs", action="store_true", help="Show parser logs")
    ap.add_argument(
        "--hf-offline", action="store_true", help="Enable HuggingFace offline mode"
    )
    ap.add_argument(
        "--type",
        type=str,
        default=None,
        choices=["lsf", "mineru", "docling"],
        help="Run only the specified parser type (default: run docling + lsf together)",
    )

    args = ap.parse_args()
    try:
        preprocess_documents(
            dataset=args.dataset,
            limit=args.limit,
            force=args.force,
            include_no_gt=args.include_no_gt,
            experiment=args.experiment,
            workers=args.workers,
            verbose_parser_logs=args.verbose_parser_logs,
            hf_offline=args.hf_offline,
            run_type=args.type,
        )
    except RuntimeError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
