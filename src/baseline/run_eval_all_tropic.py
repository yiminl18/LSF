"""Evaluate a dataset-scope baseline across all questions × docs for the tropic dataset.

This runner is for baselines that execute one global agent over a whole court
document set and a whole question set, then materialize the standard per-pair
result format afterward.
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

DATASET = "tropic"
QUERIES_FILE = _ROOT / "data/tropic/queries.json"
TEXT_DIR = _ROOT / "data/tropic/text"
LABELS_FILE = _ROOT / "data/tropic/all_labels.json"

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
Equivalence rules:
- Ignore capitalization, punctuation, and leading/trailing whitespace differences
- Treat abbreviations and full forms as equivalent
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
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    resp = gpt54_mod.client.chat.completions.create(
        model=gpt54_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: {verdict!r}")
    return verdict == "correct"


def _distribute_integer_total(total: int, n: int) -> list[int]:
    if n <= 0:
        return []
    base = total // n
    remainder = total % n
    return [base + (1 if idx < remainder else 0) for idx in range(n)]


def _default_output_name(baseline: str, split: str) -> str:
    return f"{baseline}/{split}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dataset-scope baseline eval across all court questions × docs")
    ap.add_argument(
        "--baseline",
        default="agentic_codex_qa_gpt54_all",
        help="Dataset-scope baseline module name under src/baseline/ (default: agentic_codex_qa_gpt54_all)",
    )
    ap.add_argument("--model", default="gpt54", help="Model alias forwarded to the baseline module")
    ap.add_argument("--question-slug", default=None, help="Run only questions whose slug starts with this prefix")
    ap.add_argument("--max-docs", type=int, default=None, help="Run at most N docs from sorted labels")
    ap.add_argument("--start-doc", type=int, default=0, help="0-based starting doc offset after sorting labels")
    ap.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output directory name under baseline_results/tropic/",
    )
    ap.add_argument(
        "--split-name",
        default="all_docs",
        help="Label written into records and summary rows (default: all_docs)",
    )
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    baseline_mod = importlib.import_module(f"baseline.{args.baseline}")
    if not hasattr(baseline_mod, "run_dataset"):
        raise AttributeError(f"{args.baseline} does not expose run_dataset(...)")
    gpt54_mod = importlib.import_module("models.gpt54")

    queries_raw = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    questions = [q["text"] for q in queries_raw]
    labels: dict[str, dict] = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    out_name = args.output_name or _default_output_name(args.baseline, args.split_name)
    out_base = _ROOT / "baseline_results" / DATASET / out_name
    out_base.mkdir(parents=True, exist_ok=True)

    active_slugs = _active_question_slugs(questions, args.question_slug)
    completed_docs = _completed_docs_for_questions(out_base, active_slugs)

    selected_items = sorted(labels.items())
    if args.start_doc:
        selected_items = selected_items[args.start_doc :]
    if args.skip_existing:
        selected_items = [
            (pdf_key, doc_labels)
            for pdf_key, doc_labels in selected_items
            if pdf_key.replace(".pdf", "") not in completed_docs
        ]
    if args.max_docs is not None:
        selected_items = selected_items[: args.max_docs]
    labels = dict(selected_items)

    active_questions = [
        question
        for question in questions
        if not args.question_slug or _make_slug(question).startswith(args.question_slug)
    ]

    docs_for_run: dict[str, Path] = {}
    labels_for_run: dict[str, dict] = {}
    for pdf_key, doc_labels in labels.items():
        doc_name = pdf_key.replace(".pdf", "")
        doc_path = TEXT_DIR / f"{doc_name}.txt"
        if not doc_path.exists():
            print(f"  SKIP (no txt): {doc_name}")
            continue
        docs_for_run[doc_name] = doc_path
        labels_for_run[doc_name] = doc_labels

    total_pairs = len(active_questions) * len(docs_for_run)
    metadata = {
        "baseline": args.baseline,
        "model": args.model,
        "split": args.split_name,
        "labels_file": str(LABELS_FILE.relative_to(_ROOT)),
        "output_name": out_name,
        "question_slug_prefix": args.question_slug,
        "start_doc": args.start_doc,
        "max_docs": args.max_docs,
        "selected_docs": sorted(docs_for_run),
        "completed_docs_skipped": sorted(completed_docs),
        "dataset_scope_execution": True,
        "doc_count": len(docs_for_run),
        "question_count": len(active_questions),
        "pair_count": total_pairs,
    }
    (out_base / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"baseline={args.baseline}  model={args.model}  questions={len(active_questions)}  docs={len(docs_for_run)}")
    print(f"output={out_base}")
    print(f"labels_file={LABELS_FILE}")
    if args.max_docs is not None:
        print(f"completed_docs_skipped={len(completed_docs)}  selected_docs={len(docs_for_run)}")
    print()

    run_result = baseline_mod.run_dataset(
        docs=docs_for_run,
        questions=active_questions,
        model=args.model,
        timeout=args.timeout,
        log_dir=out_base / "logs",
        run_stem=args.split_name,
        dataset_name=DATASET,
        split_name=args.split_name,
    )

    input_alloc = _distribute_integer_total(int(run_result.get("input_tokens_total", 0) or 0), total_pairs)
    output_alloc = _distribute_integer_total(int(run_result.get("output_tokens_total", 0) or 0), total_pairs)
    cached_alloc = _distribute_integer_total(int(run_result.get("cached_input_tokens_total", 0) or 0), total_pairs)
    reasoning_alloc = _distribute_integer_total(int(run_result.get("reasoning_output_tokens_total", 0) or 0), total_pairs)
    latency_per_pair = (run_result.get("latency_seconds_total", 0.0) or 0.0) / total_pairs if total_pairs else 0.0

    answers_by_doc: dict[str, dict[str, str]] = run_result.get("answers_by_doc", {})
    pair_index = 0
    all_summaries: list[dict] = []

    for question in active_questions:
        slug = _make_slug(question)
        q_dir = out_base / slug
        q_dir.mkdir(exist_ok=True)
        per_doc_results: list[dict] = []

        for doc_name in sorted(docs_for_run):
            out_file = q_dir / f"{doc_name}.json"
            doc_labels = labels_for_run[doc_name]
            answer = answers_by_doc.get(doc_name, {}).get(question)
            pair_status = run_result.get("status")
            if answer is None:
                pair_status = "missing_pair" if pair_status == "ok" else f"{pair_status}_missing_pair"

            correct = _judge(question, doc_labels.get(question), answer, gpt54_mod)
            record = {
                "doc_name": doc_name,
                "question": question,
                "question_slug": slug,
                "split": args.split_name,
                "ground_truth": doc_labels.get(question),
                "answer": answer,
                "correct": correct,
                "status": pair_status,
                "input_tokens": input_alloc[pair_index] if pair_index < len(input_alloc) else 0,
                "output_tokens": output_alloc[pair_index] if pair_index < len(output_alloc) else 0,
                "cached_input_tokens": cached_alloc[pair_index] if pair_index < len(cached_alloc) else 0,
                "reasoning_output_tokens": reasoning_alloc[pair_index] if pair_index < len(reasoning_alloc) else 0,
                "latency_seconds": latency_per_pair,
                "total_cost_usd": run_result.get("total_cost_usd"),
                "model": run_result.get("model", args.model),
                "codex_thread_id": run_result.get("codex_thread_id"),
                "codex_event_count": run_result.get("codex_event_count"),
                "codex_error_message": run_result.get("codex_error_message"),
                "codex_log_path": run_result.get("codex_log_path"),
                "codex_last_message_path": run_result.get("codex_last_message_path"),
                "stderr": run_result.get("stderr"),
                "token_accounting_scope": "run_distributed_exact",
                "run_total_input_tokens": run_result.get("input_tokens_total", 0),
                "run_total_output_tokens": run_result.get("output_tokens_total", 0),
                "run_total_cached_input_tokens": run_result.get("cached_input_tokens_total", 0),
                "run_total_reasoning_output_tokens": run_result.get("reasoning_output_tokens_total", 0),
                "run_total_latency_seconds": run_result.get("latency_seconds_total", 0.0),
                "answers_output_path": run_result.get("answers_output_path"),
                "manifest_path": run_result.get("manifest_path"),
            }
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            per_doc_results.append(record)
            pair_index += 1
            status = "✓" if correct else "✗"
            print(f"  {status} {doc_name}: {record['answer']!r}")

        if not per_doc_results:
            continue

        n = len(per_doc_results)
        n_correct = sum(r["correct"] for r in per_doc_results)
        accuracy = round(n_correct / n, 4)
        avg_in = round(mean(r["input_tokens"] for r in per_doc_results), 1)
        avg_out = round(mean(r["output_tokens"] for r in per_doc_results), 1)
        avg_lat = round(mean(r["latency_seconds"] for r in per_doc_results), 2)

        q_summary = {
            "question": question,
            "question_slug": slug,
            "split": args.split_name,
            "model": args.model,
            "n": n,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "avg_input_tokens": avg_in,
            "avg_output_tokens": avg_out,
            "avg_latency_seconds": avg_lat,
            "token_accounting_scope": "run_distributed_exact",
            "run_total_input_tokens": run_result.get("input_tokens_total", 0),
            "run_total_output_tokens": run_result.get("output_tokens_total", 0),
            "run_total_cached_input_tokens": run_result.get("cached_input_tokens_total", 0),
            "run_total_reasoning_output_tokens": run_result.get("reasoning_output_tokens_total", 0),
            "run_total_latency_seconds": run_result.get("latency_seconds_total", 0.0),
        }
        all_summaries.append(q_summary)
        print(f"\nQuestion: {question[:70]}")
        print(
            f"  accuracy={accuracy:.2f} ({n_correct}/{n})  "
            f"avg_in={avg_in:.0f}  avg_out={avg_out:.0f}  avg_lat={avg_lat:.1f}s\n"
        )

    metadata.update(
        {
            "run_status": run_result.get("status"),
            "input_tokens_total": run_result.get("input_tokens_total", 0),
            "output_tokens_total": run_result.get("output_tokens_total", 0),
            "cached_input_tokens_total": run_result.get("cached_input_tokens_total", 0),
            "reasoning_output_tokens_total": run_result.get("reasoning_output_tokens_total", 0),
            "latency_seconds_total": run_result.get("latency_seconds_total", 0.0),
            "codex_thread_id": run_result.get("codex_thread_id"),
            "codex_event_count": run_result.get("codex_event_count"),
            "codex_error_message": run_result.get("codex_error_message"),
            "codex_log_path": run_result.get("codex_log_path"),
            "codex_last_message_path": run_result.get("codex_last_message_path"),
            "answers_output_path": run_result.get("answers_output_path"),
            "manifest_path": run_result.get("manifest_path"),
            "stderr": run_result.get("stderr"),
        }
    )
    (out_base / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_path = out_base / "summary.json"
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
        print(f"\n{'=' * 60}")
        print(f"  mean accuracy : {mean(s['accuracy'] for s in all_summaries):.4f}")
        print(f"  mean in-tokens: {mean(s['avg_input_tokens'] for s in all_summaries):.0f}")
        print(f"  mean latency  : {mean(s['avg_latency_seconds'] for s in all_summaries):.1f}s")
        print(f"  results in    : {out_base}/")


if __name__ == "__main__":
    main()
