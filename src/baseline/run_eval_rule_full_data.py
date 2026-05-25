"""Run Strategy 4 rule generation on text datasets."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_FINANCE_SPLITS = {
    "sampled": _ROOT / "data/financebench/sample/multi_cluster/random/sample_doc_labels.json",
    "unsampled": _ROOT / "data/financebench/sample/multi_cluster/random/unsampled_doc_labels.json",
}

_DATASET_CONFIG = {
    "financebench": {
        "queries_file": _ROOT / "data/financebench/multi_clsuter_queries.txt",
        "text_dir": _ROOT / "data/financebench/text",
        "labels_file": None,
    },
    "court": {
        "queries_file": _ROOT / "data/court/queries.json",
        "text_dir": _ROOT / "data/court/text",
        "labels_file": _ROOT / "data/court/all_labels.json",
    },
    "nopv": {
        "queries_file": _ROOT / "data/nopv/queries.json",
        "text_dir": _ROOT / "data/nopv/text",
        "labels_file": _ROOT / "data/nopv/all_labels.json",
    },
    "officeqa": {
        "queries_file": _ROOT / "data/officeqa/queries.json",
        "text_dir": _ROOT / "data/officeqa/text",
        "labels_file": _ROOT / "data/officeqa/all_labels.json",
    },
}


def _make_slug(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _load_questions(dataset: str) -> list[str]:
    path = _DATASET_CONFIG[dataset]["queries_file"]
    if path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["text"] for item in data]


def _load_labels(dataset: str, split: str) -> dict[str, dict]:
    if dataset != "financebench":
        path = _DATASET_CONFIG[dataset]["labels_file"]
        return json.loads(path.read_text(encoding="utf-8"))

    if split != "all_docs":
        path = _FINANCE_SPLITS[split]
        return json.loads(path.read_text(encoding="utf-8"))

    merged: dict[str, dict] = {}
    for path in (_FINANCE_SPLITS["sampled"], _FINANCE_SPLITS["unsampled"]):
        data = json.loads(path.read_text(encoding="utf-8"))
        for pdf_key, doc_labels in data.items():
            if pdf_key in merged and merged[pdf_key] != doc_labels:
                raise ValueError(f"Conflicting labels for duplicated doc: {pdf_key}")
            merged[pdf_key] = doc_labels
    return merged


def _validate_financebench_questions(
    questions: list[str],
    labels: dict[str, dict],
) -> None:
    if not labels:
        return
    label_questions = list(next(iter(labels.values())).keys())
    if questions != label_questions:
        raise ValueError(
            "FinanceBench query file does not match label question order.\n"
            f"queries_file has {len(questions)} questions but labels have {len(label_questions)}.\n"
            f"queries_file: {questions}\n"
            f"labels: {label_questions}"
        )


def _select_docs(
    dataset: str,
    labels: dict[str, dict],
    *,
    start_doc: int,
    max_docs: int | None,
    latest_docs: bool,
) -> dict[str, dict]:
    items = sorted(labels.items())
    if latest_docs:
        if max_docs is None:
            raise SystemExit("--latest-docs requires --max-docs")
        items = items[-max_docs:]
    else:
        if start_doc:
            items = items[start_doc:]
        if max_docs is not None:
            items = items[:max_docs]
    return dict(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Strategy 4 rule generation on a text dataset")
    ap.add_argument("--dataset", choices=sorted(_DATASET_CONFIG), required=True)
    ap.add_argument(
        "--baseline",
        default="agentic_rule_full_data_gpt54",
        help="Rule-generation module under src/baseline/ (default: agentic_rule_full_data_gpt54)",
    )
    ap.add_argument("--model", default="gpt54", help="Model alias forwarded to the baseline module")
    ap.add_argument("--split", default="all_docs", help="Logical split label (default: all_docs)")
    ap.add_argument("--split-name", default=None, help="Optional output split label override")
    ap.add_argument("--question-slug", default=None, help="Run only questions whose slug starts with this prefix")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--start-doc", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--latest-docs", action="store_true")
    ap.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output name under both rules/<dataset>/ and results/<dataset>/",
    )
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    baseline_mod = importlib.import_module(f"baseline.{args.baseline}")
    runner_fn = getattr(baseline_mod, "run_rule_gen", None) or getattr(baseline_mod, "run_dataset", None)
    if runner_fn is None:
        raise AttributeError(f"{args.baseline} does not expose run_rule_gen(...)")

    split_name = args.split_name or args.split
    questions_all = _load_questions(args.dataset)
    labels_all = _load_labels(args.dataset, args.split)
    if args.dataset == "financebench":
        _validate_financebench_questions(questions_all, labels_all)
    labels_selected = _select_docs(
        args.dataset,
        labels_all,
        start_doc=args.start_doc,
        max_docs=args.max_docs,
        latest_docs=args.latest_docs,
    )

    text_dir = _DATASET_CONFIG[args.dataset]["text_dir"]
    docs_for_run: dict[str, Path] = {}
    labels_by_doc: dict[str, dict] = {}
    for pdf_key, doc_labels in sorted(labels_selected.items()):
        doc_name = pdf_key.replace(".pdf", "")
        doc_path = text_dir / f"{doc_name}.txt"
        if not doc_path.exists():
            print(f"SKIP (no txt): {doc_name}")
            continue
        docs_for_run[doc_name] = doc_path
        labels_by_doc[doc_name] = doc_labels

    active_questions = [
        question
        for question in questions_all
        if not args.question_slug or _make_slug(question).startswith(args.question_slug)
    ]
    if args.max_questions is not None:
        active_questions = active_questions[: args.max_questions]

    base_name = args.output_name or f"{args.baseline}/{split_name}"
    rules_root = _ROOT / "rules" / args.dataset / base_name
    results_root = _ROOT / "results" / args.dataset / base_name
    rules_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    print(
        f"dataset={args.dataset} baseline={args.baseline} model={args.model} "
        f"questions={len(active_questions)} docs={len(docs_for_run)}"
    )
    print(f"rules_root={rules_root}")
    print(f"results_root={results_root}\n")

    q_index = 0
    for question in active_questions:
        q_index += 1
        question_slug = _make_slug(question)
        q_token = f"q{q_index:02d}_{question_slug}"
        q_rules_dir = rules_root / q_token
        q_report_path = results_root / f"{q_token}_rule_gen.json"
        if args.skip_existing and q_report_path.exists():
            print(f"SKIP existing {q_token}")
            continue

        print(f"[q{q_index:02d}] {question}")
        result = runner_fn(
            docs=docs_for_run,
            questions=[question],
            labels_by_doc=labels_by_doc,
            model=args.model,
            timeout=args.timeout,
            dataset_name=args.dataset,
            split_name=split_name,
            rules_dir=q_rules_dir,
            results_dir=results_root,
            run_stem=f"q{q_index:02d}",
        )
        print(
            f"  status={result.get('status')}  "
            f"rules={result.get('num_rules_final')}  "
            f"match_rate_sample={result.get('rule_gen', {}).get('match_rate_sample')}  "
            f"avg_cost_ratio_sample={result.get('rule_gen', {}).get('avg_cost_ratio_sample')}\n"
        )


if __name__ == "__main__":
    main()
