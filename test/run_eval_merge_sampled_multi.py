"""Evaluate merged LLM rules on 18 sampled multi-cluster docs for every mix query."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge
import importlib
model_mod = importlib.import_module("models.gpt54")

QUERIES_FILE   = "data/financebench/mix_doc_queries.txt"
LABELS_FILE    = "data/financebench/sample/multi_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_BASE_DIR = "rules/financebench_multi_clusters/llm/gpt54/one_shot"
MERGE_RUN_DIR  = "results/financebench_multi_clusters/llm/gpt54/one_shot/rule_run_merge"
OUTPUT_DIR     = "results/financebench_multi_clusters/llm/gpt54/one_shot/eval_merge"

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
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def count_tokens(text: str) -> int:
    return len(text.split())


def judge(question: str, ground_truth, predicted) -> bool:
    if ground_truth is None:
        return False
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
        warnings.warn(f"Unexpected judge response: '{verdict}'")
    return verdict == "correct"


# ── Load questions ─────────────────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
print(f"Questions: {len(questions)}")

# ── Load labels ────────────────────────────────────────────────────────────────
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))

# ── Load documents ─────────────────────────────────────────────────────────────
doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")
print(f"Documents loaded: {len(doc_map)}\n")

# ── Precompute total token counts per doc ──────────────────────────────────────
doc_total_tokens: dict[str, int] = {
    doc_name: count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
    for doc_name, doc in doc_map.items()
}

# ── Ensure output dirs ─────────────────────────────────────────────────────────
Path(MERGE_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

summary: list[dict] = []

for question in questions:
    slug          = make_slug(question)
    question_slug = f"{slug}_18"
    rule_slug     = f"{slug}_18_llm"
    rule_folder   = Path(RULES_BASE_DIR) / rule_slug
    out_file      = Path(OUTPUT_DIR) / f"{question_slug}_sampled.json"

    if out_file.exists():
        print(f"SKIP (exists): {out_file}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append({
            "question": existing["question"],
            "question_slug": existing["question_slug"],
            "sampled": {k: v for k, v in existing.items()
                        if k not in ("per_doc", "question", "question_slug")},
        })
        continue

    if not rule_folder.is_dir():
        print(f"SKIP (no rule folder): {rule_folder}")
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"SKIP (empty rule folder): {rule_folder}")
        continue

    print(f"\nQuestion: {question}  ({len(rule_names)} rules)")

    try:
        run_results: list[dict] = []

        for doc_name, document in doc_map.items():
            try:
                result = rule_apply_merge(
                    document=document,
                    rule_names=rule_names,
                    question_slug=rule_slug,
                    question=question,
                    rules_dir=RULES_BASE_DIR,
                    output_dir=MERGE_RUN_DIR,
                )
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        result["predicted_answer"],
                    "latency_seconds":  result["latency_seconds"],
                    "retrieved_tokens": result["retrieved_token_count"],
                    "input_tokens":     result["input_tokens"],
                })
            except Exception as e:
                print(f"  ERROR {doc_name}: {e}")
                run_results.append({
                    "doc_name":         doc_name,
                    "predicted":        None,
                    "latency_seconds":  0.0,
                    "retrieved_tokens": 0,
                    "input_tokens":     0,
                })

        # ── Judge each prediction ──────────────────────────────────────────────
        per_doc_out: list[dict] = []
        cost_ratios: list[float] = []

        for r in run_results:
            doc_name     = r["doc_name"]
            ground_truth = labels.get(doc_name + ".pdf", {}).get(question)
            correct      = judge(question, ground_truth, r["predicted"])
            total_tok    = doc_total_tokens.get(doc_name, 1)
            cost_ratio   = r["retrieved_tokens"] / total_tok if total_tok > 0 else 0.0
            cost_ratios.append(cost_ratio)

            per_doc_out.append({
                "doc_name":         doc_name,
                "predicted":        r["predicted"],
                "ground_truth":     ground_truth,
                "correct":          correct,
                "latency_seconds":  r["latency_seconds"],
                "retrieved_tokens": r["retrieved_tokens"],
                "input_tokens":     r["input_tokens"],
            })
            status = "✓" if correct else "✗"
            print(f"  {status} {doc_name:<45}  pred={str(r['predicted'])[:30]!r}")

        # ── Aggregate ─────────────────────────────────────────────────────────
        n              = len(per_doc_out)
        n_correct      = sum(r["correct"] for r in per_doc_out)
        accuracy       = round(n_correct / n, 4) if n else 0.0
        avg_latency    = round(mean(r["latency_seconds"]  for r in per_doc_out), 3)
        avg_retrieved  = round(mean(r["retrieved_tokens"] for r in per_doc_out), 1)
        avg_input_tok  = round(mean(r["input_tokens"]     for r in per_doc_out), 1)
        avg_cost_ratio = round(mean(cost_ratios), 6)

        print(f"  accuracy={accuracy:.2f}  ({n_correct}/{n})  avg_cost_ratio={avg_cost_ratio:.5f}")

        per_q = {
            "question":       question,
            "question_slug":  question_slug,
            "split":          "sampled",
            "n":              n,
            "n_correct":      n_correct,
            "accuracy":       accuracy,
            "avg_latency":    avg_latency,
            "avg_retrieved":  avg_retrieved,
            "avg_input_tok":  avg_input_tok,
            "avg_cost_ratio": avg_cost_ratio,
            "per_doc":        per_doc_out,
        }
        out_file.write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

        summary.append({
            "question":      question,
            "question_slug": question_slug,
            "sampled": {
                "question":       question,
                "question_slug":  question_slug,
                "split":          "sampled",
                "n":              n,
                "n_correct":      n_correct,
                "accuracy":       accuracy,
                "avg_latency":    avg_latency,
                "avg_retrieved":  avg_retrieved,
                "avg_input_tok":  avg_input_tok,
                "avg_cost_ratio": avg_cost_ratio,
            },
        })

    except Exception as e:
        import traceback
        print(f"  FATAL ERROR for '{question}': {e}")
        traceback.print_exc()
        continue

# ── Write summary ──────────────────────────────────────────────────────────────
summary_path = Path(OUTPUT_DIR) / "summary_sampled.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

# ── Print table ────────────────────────────────────────────────────────────────
print(f"\n{'Question':<55} {'Acc':>6}  {'AvgTok':>8}  {'CostRatio':>10}")
print("-" * 85)
for s in summary:
    sq = s["sampled"]
    print(
        f"  {s['question'][:53]:<53}  {sq['accuracy']:>5.2f}  "
        f"{sq['avg_retrieved']:>8.1f}  {sq['avg_cost_ratio']:>10.5f}"
    )

print(f"\nResults written to: {OUTPUT_DIR}/")
print(f"Summary:            {summary_path}")
