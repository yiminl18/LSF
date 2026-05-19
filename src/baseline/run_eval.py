"""Evaluate a baseline strategy across all questions × all docs in a split.

Output layout:
    baseline_results/<dataset>/<baseline>_<model>/
        <question_slug>/<doc_name>.json    # one file per (question, doc) pair
        summary.json                        # mean accuracy, tokens, latency

Usage:
    # Agentic Claude QA, opus47, sampled docs
    python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled

    # GPT-54 direct, unsampled docs
    python src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54 --split unsampled

    # Single question (by slug prefix)
    python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled \
        --question-slug what_is_the_registrants_telephone_number
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

DATASET         = "financebench"
QUERIES_FILE    = _ROOT / "data/financebench/sample_queries.txt"
PROCESSING_DIR  = _ROOT / "data/financebench/processing"

_SPLIT_LABELS = {
    "sampled":   _ROOT / "data/financebench/sample/single_cluster/random/sample_doc_labels.json",
    "unsampled": _ROOT / "data/financebench/sample/single_cluster/random/unsampled_doc_labels.json",
}

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT_FOUND", "NOT FOUND", or null, always judge as incorrect
Reply with exactly one word: CORRECT or INCORRECT"""


def _make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def _active_question_slugs(questions: list[str], question_slug_prefix: str | None) -> list[str]:
    slugs: list[str] = []
    for question in questions:
        slug = _make_slug(question)
        if question_slug_prefix and not slug.startswith(question_slug_prefix):
            continue
        slugs.append(slug)
    return slugs


def _completed_docs_for_questions(out_base: Path, question_slugs: list[str]) -> set[str]:
    if not question_slugs:
        return set()

    per_question_docs: list[set[str]] = []
    for slug in question_slugs:
        q_dir = out_base / slug
        docs = {p.stem for p in q_dir.glob("*.json")} if q_dir.exists() else set()
        per_question_docs.append(docs)

    return set.intersection(*per_question_docs) if per_question_docs else set()


def _judge(question: str, ground_truth, predicted, gpt54_mod) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {verdict!r}")
    return verdict == "correct"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline eval across all questions × docs")
    ap.add_argument("--baseline", default="agentic_claude_qa",
                    help="Baseline module name under src/baseline/ (default: agentic_claude_qa)")
    ap.add_argument("--model",    default="opus47",
                    help="Model alias (default: opus47)")
    ap.add_argument("--split",    choices=("sampled", "unsampled"), default="sampled")
    ap.add_argument("--labels-file", default=None,
                    help="Optional path to a labels JSON file; overrides --split")
    ap.add_argument("--question-slug", default=None,
                    help="Run only questions whose slug starts with this prefix")
    ap.add_argument("--output-name", default=None,
                    help="Optional custom output directory name under baseline_results/<dataset>/")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Run at most N not-yet-completed docs from the selected labels set")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--timeout",  type=int, default=300)
    args = ap.parse_args()

    baseline_mod = importlib.import_module(f"baseline.{args.baseline}")
    gpt54_mod    = importlib.import_module("models.gpt54")

    questions    = [l.strip() for l in QUERIES_FILE.read_text().splitlines() if l.strip()]
    labels_path  = Path(args.labels_file) if args.labels_file else _SPLIT_LABELS[args.split]
    labels: dict[str, dict] = json.loads(labels_path.read_text(encoding="utf-8"))

    out_name = args.output_name or f"{args.baseline}_{args.model}"
    out_base = _ROOT / "baseline_results" / DATASET / out_name
    out_base.mkdir(parents=True, exist_ok=True)

    active_slugs = _active_question_slugs(questions, args.question_slug)
    completed_docs = _completed_docs_for_questions(out_base, active_slugs)

    selected_items = sorted(labels.items())
    if args.max_docs is not None:
        selected_items = [
            (pdf_key, doc_labels)
            for pdf_key, doc_labels in selected_items
            if pdf_key.replace(".pdf", "") not in completed_docs
        ][:args.max_docs]
        labels = dict(selected_items)

    if args.max_docs is not None:
        metadata = {
            "baseline": args.baseline,
            "model": args.model,
            "split": args.split,
            "labels_file": str(labels_path.relative_to(_ROOT)) if labels_path.is_relative_to(_ROOT) else str(labels_path),
            "output_name": out_name,
            "question_slug_prefix": args.question_slug,
            "max_docs": args.max_docs,
            "selected_docs": [pdf_key.replace(".pdf", "") for pdf_key, _ in selected_items],
            "completed_docs_skipped": sorted(completed_docs),
        }
        (out_base / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"baseline={args.baseline}  model={args.model}  split={args.split}")
    print(f"questions={len(questions)}  docs={len(labels)}  output={out_base}")
    print(f"labels_file={labels_path}")
    if args.max_docs is not None:
        print(f"completed_docs_skipped={len(completed_docs)}  selected_docs={len(labels)}")
    print()

    all_summaries: list[dict] = []

    for question in questions:
        slug = _make_slug(question)
        if args.question_slug and not slug.startswith(args.question_slug):
            continue

        q_dir = out_base / slug
        q_dir.mkdir(exist_ok=True)

        per_doc_results: list[dict] = []

        for pdf_key, doc_labels in labels.items():
            doc_name  = pdf_key.replace(".pdf", "")
            doc_path  = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
            out_file  = q_dir / f"{doc_name}.json"

            if args.skip_existing and out_file.exists():
                try:
                    per_doc_results.append(json.loads(out_file.read_text(encoding="utf-8")))
                except Exception:
                    pass
                continue

            if not doc_path.exists():
                print(f"  SKIP (no JSON): {doc_name}")
                continue

            ground_truth = doc_labels.get(question)

            try:
                result = baseline_mod.run_qa(
                    doc_path=doc_path,
                    question=question,
                    model=args.model,
                    timeout=args.timeout,
                    log_dir=q_dir / "logs",
                    log_stem=doc_name,
                )
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                result = {
                    "status": "error",
                    "answer": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_seconds": 0.0,
                    "total_cost_usd": None,
                    "model": args.model,
                }

            correct = _judge(question, ground_truth, result.get("answer"), gpt54_mod)

            record = {
                "doc_name":        doc_name,
                "question":        question,
                "question_slug":   slug,
                "split":           args.split,
                "ground_truth":    ground_truth,
                "answer":          result.get("answer"),
                "correct":         correct,
                "status":          result.get("status"),
                "input_tokens":    result.get("input_tokens", 0),
                "output_tokens":   result.get("output_tokens", 0),
                "latency_seconds": result.get("latency_seconds", 0.0),
                "total_cost_usd":  result.get("total_cost_usd"),
                "model":           result.get("model", args.model),
            }
            for k, v in result.items():
                if k not in record and k != "answer":
                    record[k] = v
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            per_doc_results.append(record)
            status = "✓" if correct else "✗"
            print(f"  {status} {doc_name}: {record['answer']!r}")

        if not per_doc_results:
            continue

        n         = len(per_doc_results)
        n_correct = sum(r["correct"] for r in per_doc_results)
        accuracy  = round(n_correct / n, 4)
        avg_in    = round(mean(r["input_tokens"]    for r in per_doc_results), 1)
        avg_out   = round(mean(r["output_tokens"]   for r in per_doc_results), 1)
        avg_lat   = round(mean(r["latency_seconds"] for r in per_doc_results), 2)

        q_summary = {
            "question":        question,
            "question_slug":   slug,
            "split":           args.split,
            "model":           args.model,
            "n":               n,
            "n_correct":       n_correct,
            "accuracy":        accuracy,
            "avg_input_tokens":  avg_in,
            "avg_output_tokens": avg_out,
            "avg_latency_seconds": avg_lat,
        }
        all_summaries.append(q_summary)
        print(f"\nQuestion: {question[:70]}")
        print(f"  accuracy={accuracy:.2f} ({n_correct}/{n})  "
              f"avg_in={avg_in:.0f}  avg_out={avg_out:.0f}  avg_lat={avg_lat:.1f}s\n")

    summary_path = out_base / "summary.json"
    # Merge with any existing summary entries for other splits/runs
    existing: list[dict] = []
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    merged = {e["question_slug"]: e for e in existing}
    for s in all_summaries:
        merged[s["question_slug"]] = s
    summary_path.write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_summaries:
        print(f"\n{'='*60}")
        print(f"  mean accuracy : {mean(s['accuracy'] for s in all_summaries):.4f}")
        print(f"  mean in-tokens: {mean(s['avg_input_tokens'] for s in all_summaries):.0f}")
        print(f"  mean latency  : {mean(s['avg_latency_seconds'] for s in all_summaries):.1f}s")
        print(f"  results in    : {out_base}/")


if __name__ == "__main__":
    main()
