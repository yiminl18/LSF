"""Rule-generation entrypoint for Strategy 4 with GPT-5.4-mini Codex."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from baseline.agentic_rule_full_data import run_rule_gen as _run_rule_gen


def run_rule_gen(
    docs: dict[str, str | Path],
    questions: list[str],
    *,
    labels_by_doc: dict[str, dict[str, Any]] | None = None,
    model: str = "gpt54mini",
    timeout: int = 3600,
    dataset_name: str = "court",
    split_name: str = "all_docs",
    rules_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    run_stem: str = "q01",
    **kwargs: Any,
) -> dict[str, Any]:
    return _run_rule_gen(
        docs=docs,
        questions=questions,
        labels_by_doc=labels_by_doc,
        model=model,
        timeout=timeout,
        dataset_name=dataset_name,
        split_name=split_name,
        rules_dir=rules_dir,
        results_dir=results_dir,
        run_stem=run_stem,
        **kwargs,
    )


def run_dataset(
    docs: dict[str, str | Path],
    questions: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    return run_rule_gen(docs=docs, questions=questions, **kwargs)
