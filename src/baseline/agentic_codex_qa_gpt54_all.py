"""Baseline entrypoint for dataset-scope Agentic Codex QA with GPT-5.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baseline.agentic_codex_qa_all import run_dataset as _run_dataset


def run_dataset(
    docs: dict[str, str | Path],
    questions: list[str],
    *,
    model: str = "gpt54",
    timeout: int = 3600,
    log_dir: str | Path | None = None,
    run_stem: str = "all_docs",
    dataset_name: str = "financebench",
    split_name: str = "all_docs",
    **_: Any,
) -> dict[str, Any]:
    return _run_dataset(
        docs=docs,
        questions=questions,
        model=model,
        timeout=timeout,
        log_dir=log_dir,
        run_stem=run_stem,
        dataset_name=dataset_name,
        split_name=split_name,
    )
