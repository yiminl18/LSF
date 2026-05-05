"""Evaluate merge results for all questions on both sampled and unsampled docs.

Per-question results are saved immediately after each question completes so
the run can be restarted without re-judging already-evaluated questions.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

QUERIES_FILE      = "data/financebench/sample_queries.txt"
SAMPLE_LABELS     = "data/financebench/sample_doc_labels.json"
UNSAMPLED_LABELS  = "data/financebench/unsampled_doc_labels.json"
RULES_DIR         = "rules/financebench"
SAMPLED_RUN_DIR   = "results/financebench/rule_run/merge"
UNSAMPLED_RUN_DIR = "results/financebench/rule_run/merge_unsampled"
EVAL_OUT_DIR      = "results/financebench/eval_merge_all"

Path(EVAL_OUT_DIR).mkdir(parents=True, exist_ok=True)

model_mod = importlib.import_module("models.gpt54")

_JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""


def make_slug(q: str) -> str:
    s = re.sub(r"[^\w\s]", "", q.lower())
    return re.sub(r"\s+", "_", s)[:60]


def llm_judge(question: str, ground_truth, predicted) -> bool:
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted) if predicted is not None else "null"
    user_msg = f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge verdict: '{verdict}'")
    return verdict == "correct"


def evaluate_split(
    run_dir: str,
    question_slug: str,
    question: str,
    labels: dict,
    split_tag: str,
) -> dict | None:
    # Skip if already evaluated
    out_path = Path(EVAL_OUT_DIR) / f"{question_slug}_{split_tag}.json"
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"    [SKIP — already done] accuracy={data['accuracy']:.2f}  avg_lat={data['avg_latency']:.2f}s")
        return data

    q_dir = Path(run_dir) / question_slug
    if not q_dir.exists():
        return None
    files = list(q_dir.glob("*_merge.json"))
    if not files:
        return None

    records = json.loads(files[0].read_text(encoding="utf-8"))
    records_by_doc = {r["doc_name"]: r for r in records}

    per_doc = []
    for pdf_key, qa in labels.items():
        doc_name     = pdf_key.replace(".pdf", "")
        ground_truth = qa.get(question)
        pred_record  = records_by_doc.get(doc_name)
        if pred_record is None:
            continue

        predicted        = pred_record["predicted_answer"]
        latency          = pred_record["latency_seconds"]
        retrieved_tokens = pred_record["retrieved_token_count"]
        input_tokens     = pred_record["input_tokens"]

        correct = False if ground_truth is None else llm_judge(question, ground_truth, predicted)

        mark = "✓" if correct else "✗"
        print(f"    {mark} {doc_name:<45}  gt={str(ground_truth)[:20]:<20}  pred={str(predicted)[:20]}")

        per_doc.append({
            "doc_name":        doc_name,
            "predicted":       predicted,
            "ground_truth":    ground_truth,
            "correct":         correct,
            "latency_seconds": latency,
            "retrieved_tokens":retrieved_tokens,
            "input_tokens":    input_tokens,
        })

    if not per_doc:
        return None

    n          = len(per_doc)
    n_correct  = sum(1 for r in per_doc if r["correct"])
    accuracy   = n_correct / n
    avg_lat    = sum(r["latency_seconds"] for r in per_doc) / n
    avg_ret    = sum(r["retrieved_tokens"] for r in per_doc) / n
    avg_input  = sum(r["input_tokens"] for r in per_doc) / n
    # cost_ratio: avg retrieved tokens relative to avg input tokens sent to LLM
    avg_cost_ratio = avg_ret / avg_input if avg_input > 0 else 0.0

    result = {
        "question":      question,
        "question_slug": question_slug,
        "split":         split_tag,
        "n":             n,
        "n_correct":     n_correct,
        "accuracy":      round(accuracy, 4),
        "avg_latency":   round(avg_lat, 3),
        "avg_retrieved": round(avg_ret, 1),
        "avg_input_tok": round(avg_input, 1),
        "avg_cost_ratio":round(avg_cost_ratio, 6),
        "per_doc":       per_doc,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    Saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
questions       = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
sample_labels   = json.loads(Path(SAMPLE_LABELS).read_text(encoding="utf-8"))
unsampled_labels= json.loads(Path(UNSAMPLED_LABELS).read_text(encoding="utf-8"))

print(f"Questions: {len(questions)}  |  sampled: {len(sample_labels)}  |  unsampled: {len(unsampled_labels)}")

summary_rows = []

for question in questions:
    slug          = make_slug(question)
    question_slug = f"{slug}_10"
    print(f"\n{'='*70}")
    print(f"Question: {question}")

    row = {"question": question, "slug": question_slug}

    for split_tag, run_dir, labels in [
        ("sampled",   SAMPLED_RUN_DIR,   sample_labels),
        ("unsampled", UNSAMPLED_RUN_DIR, unsampled_labels),
    ]:
        print(f"\n  [{split_tag}]")
        try:
            metrics = evaluate_split(run_dir, question_slug, question, labels, split_tag)
        except Exception as e:
            print(f"    ERROR: {e}")
            metrics = None
        row[split_tag] = metrics
        if metrics:
            print(f"  → acc={metrics['accuracy']:.2f} ({metrics['n_correct']}/{metrics['n']})  "
                  f"lat={metrics['avg_latency']:.2f}s  cost_ratio={metrics['avg_cost_ratio']:.5f}")

    summary_rows.append(row)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 110)
print("=== SUMMARY ===")
print("=" * 110)
print(f"{'Question':<54} | {'Samp Acc':>8} {'Samp Lat':>9} {'Samp Cost':>10} | {'Unsamp Acc':>10} {'Unsamp Lat':>11} {'Unsamp Cost':>12}")
print("-" * 110)

for row in summary_rows:
    q = row["question"][:53]
    s = row.get("sampled")
    u = row.get("unsampled")
    s_acc  = f"{s['accuracy']:.2f}"       if s else "N/A"
    s_lat  = f"{s['avg_latency']:.2f}s"   if s else "N/A"
    s_cost = f"{s['avg_cost_ratio']:.5f}" if s else "N/A"
    u_acc  = f"{u['accuracy']:.2f}"       if u else "N/A"
    u_lat  = f"{u['avg_latency']:.2f}s"   if u else "N/A"
    u_cost = f"{u['avg_cost_ratio']:.5f}" if u else "N/A"
    print(f"{q:<54} | {s_acc:>8} {s_lat:>9} {s_cost:>10} | {u_acc:>10} {u_lat:>11} {u_cost:>12}")

# ---------------------------------------------------------------------------
# Save aggregate summary
# ---------------------------------------------------------------------------
agg = []
for row in summary_rows:
    agg.append({
        "question":      row["question"],
        "question_slug": row["slug"],
        "sampled":   {k: v for k, v in (row["sampled"] or {}).items() if k != "per_doc"},
        "unsampled": {k: v for k, v in (row["unsampled"] or {}).items() if k != "per_doc"},
    })

agg_path = Path(EVAL_OUT_DIR) / "summary.json"
agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nAggregate summary saved to {agg_path}")
