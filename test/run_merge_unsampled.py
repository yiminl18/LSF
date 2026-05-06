"""Apply rule_apply_merge on all 50 unsampled docs and evaluate accuracy + cost."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge

# ---------------------------------------------------------------------------
# STEP 1 — Setup
# ---------------------------------------------------------------------------
QUESTION       = "What is the registrant's exact name?"
QUESTION_SLUG  = "what_is_the_registrants_exact_name_10"
RULES_DIR      = "rules/financebench_single_cluster/llm/gpt54/one_shot"
LABELS_FILE    = "data/financebench/unsampled_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
OUTPUT_DIR     = "results/financebench_single_cluster/llm/gpt54/one_shot/rule_run_merge_unsampled"
EVAL_OUTPUT_DIR = "results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge"

# ---------------------------------------------------------------------------
# STEP 2 — Load rule names
# ---------------------------------------------------------------------------
rule_names = sorted(
    os.path.splitext(f)[0]
    for f in os.listdir(f"{RULES_DIR}/{QUESTION_SLUG}")
    if f.startswith("rule_") and f.endswith(".py")
)
print(f"Rules loaded: {len(rule_names)}")

# ---------------------------------------------------------------------------
# STEP 3 — Load unsampled labels
# ---------------------------------------------------------------------------
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names = [k.replace(".pdf", "") for k in labels.keys()]
print(f"Docs to process: {len(doc_names)}")
print()

# ---------------------------------------------------------------------------
# STEP 4 — Apply rule_apply_merge for each doc
# ---------------------------------------------------------------------------
for i, doc_name in enumerate(doc_names, 1):
    doc_path = f"{PROCESSING_DIR}/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        print(f"[{i:02d}] WARNING: not found, skipping — {doc_path}")
        continue
    document = json.loads(Path(doc_path).read_text(encoding="utf-8"))
    result = rule_apply_merge(
        document=document,
        rule_names=rule_names,
        question_slug=QUESTION_SLUG,
        question=QUESTION,
        output_dir=OUTPUT_DIR,
    )
    print(f"[{i:02d}] {doc_name:<50}  pred={str(result['predicted_answer'])[:30]}")

# ---------------------------------------------------------------------------
# STEP 5 — Load results and evaluate with LLM judge
# ---------------------------------------------------------------------------
rule_set_slug = "__".join(sorted(rule_names))[:120]
results_file = Path(OUTPUT_DIR) / QUESTION_SLUG / f"{rule_set_slug}_merge.json"
records: list[dict] = json.loads(results_file.read_text(encoding="utf-8"))
records_by_doc = {r["doc_name"]: r for r in records}

# Judge setup
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

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)

print()
print("Evaluating with LLM judge...")
per_document = []
total_judge_input_tokens = 0
total_judge_output_tokens = 0

for doc_name in doc_names:
    doc_path = f"{PROCESSING_DIR}/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        continue

    pred_record = records_by_doc.get(doc_name)
    predicted_answer = pred_record["predicted_answer"] if pred_record else None
    retrieved_token_count = pred_record["retrieved_token_count"] if pred_record else 0
    rule_apply_latency = pred_record["latency_seconds"] if pred_record else 0.0

    ground_truth = labels.get(doc_name + ".pdf", {}).get(QUESTION)

    # Total doc tokens
    document = json.loads(Path(doc_path).read_text(encoding="utf-8"))
    total_doc_tokens = _count_tokens("\n".join(s.get("text", "") for s in document.get("texts", [])))
    cost_ratio = retrieved_token_count / total_doc_tokens if total_doc_tokens > 0 else None

    # Judge
    if ground_truth is None:
        correct = False
        judge_latency = 0.0
        judge_input_tokens = judge_output_tokens = 0
    else:
        gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
        pred_str = str(predicted_answer) if predicted_answer is not None else "null"
        user_prompt = f"Question: {QUESTION}\nGround Truth: {gt_str}\nPredicted: {pred_str}"

        t_judge = time.time()
        response = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
        judge_latency = time.time() - t_judge
        judge_raw = (response.choices[0].message.content or "").strip()
        usage = response.usage
        judge_input_tokens = usage.prompt_tokens if usage else 0
        judge_output_tokens = usage.completion_tokens if usage else 0
        verdict = judge_raw.strip().lower()
        if verdict not in ("correct", "incorrect"):
            warnings.warn(f"Unexpected judge response for {doc_name}: '{judge_raw}'")
        correct = verdict == "correct"

    total_judge_input_tokens += judge_input_tokens
    total_judge_output_tokens += judge_output_tokens

    mark = "✓" if correct else "✗"
    print(f"  {mark} {doc_name:<50}  gt={str(ground_truth)[:25]:<25}  pred={str(predicted_answer)[:25]}")

    per_document.append({
        "doc_name": doc_name,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "correct": correct,
        "retrieved_token_count": retrieved_token_count,
        "total_doc_tokens": total_doc_tokens,
        "cost_ratio": round(cost_ratio, 6) if cost_ratio is not None else None,
        "judge_latency_seconds": round(judge_latency, 3),
        "rule_apply_latency_seconds": round(rule_apply_latency, 3),
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
    })

# ---------------------------------------------------------------------------
# STEP 6 — Aggregate metrics
# ---------------------------------------------------------------------------
n = len(per_document)
num_correct = sum(1 for r in per_document if r["correct"])
accuracy = num_correct / n if n > 0 else 0.0

valid_costs = [r["cost_ratio"] for r in per_document if r["cost_ratio"] is not None]
avg_cost_ratio = sum(valid_costs) / len(valid_costs) if valid_costs else 0.0

latencies = [r["rule_apply_latency_seconds"] for r in per_document]
avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
total_latency = sum(latencies)

avg_retrieved = sum(r["retrieved_token_count"] for r in per_document) / n if n > 0 else 0.0

print()
print("=" * 55)
print("=== Merge Rule Results (Unsampled, N=50) ===")
print("=" * 55)
print(f"Question:        {QUESTION}")
print(f"Num rules used:  {len(rule_names)}")
print(f"Accuracy:        {accuracy:.2f}  ({num_correct}/{n})")
print(f"Avg cost ratio:  {avg_cost_ratio:.5f}")
print(f"Avg retrieved:   {avg_retrieved:.1f} tokens")
print(f"Avg latency (s): {avg_latency:.2f}")
print(f"Total latency:   {total_latency:.1f} s")
print(f"Judge tokens:    {total_judge_input_tokens:,} in / {total_judge_output_tokens:,} out")

# ---------------------------------------------------------------------------
# STEP 7 — Write eval results
# ---------------------------------------------------------------------------
eval_result = {
    "question": QUESTION,
    "question_slug": QUESTION_SLUG,
    "strategy": "merge",
    "num_documents": n,
    "num_rules": len(rule_names),
    "rule_names": rule_names,
    "accuracy": round(accuracy, 4),
    "avg_cost_ratio": round(avg_cost_ratio, 6),
    "avg_retrieved_token_count": round(avg_retrieved, 2),
    "avg_latency_seconds": round(avg_latency, 3),
    "total_latency_seconds": round(total_latency, 3),
    "total_judge_input_tokens": total_judge_input_tokens,
    "total_judge_output_tokens": total_judge_output_tokens,
    "per_document": per_document,
}

eval_out = Path(EVAL_OUTPUT_DIR) / QUESTION_SLUG / "all_rules_merge_unsampled_eval.json"
eval_out.parent.mkdir(parents=True, exist_ok=True)
eval_out.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nEval written to {eval_out}")
