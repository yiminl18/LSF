"""Debug run: apply q03 revenue rule on 50 FinanceBench docs.

Results stored in analysis/financebench_q03_debug/results/.
Does NOT touch results/financebench/... (existing results are safe).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from rule_apply_merge import rule_apply_merge
import baseline.run_eval_rule_apply_merge_text as runner_mod

# ── Config ────────────────────────────────────────────────────────────────────
DATASET         = "financebench"
RULE_GEN_NAME   = "agentic_rule_full_data_gpt54mini/all_docs"
QUESTION        = "What is total revenue for the most recent fiscal year, from the income statement?"
QUESTION_SLUG   = "what_is_total_revenue_for_the_most_recent_fiscal_year_from_t"
RULE_QUESTION_DIR = "q03_what_is_total_revenue_for_the_most_recent_fiscal_year_from_t"
RULE_NAME       = "rule_income_statement_revenue_window"
MAX_DOCS        = 50
MODEL           = "gpt54"

TEXT_DIR  = _ROOT / "data" / DATASET / "text"
RULES_DIR = _ROOT / "rules" / DATASET / RULE_GEN_NAME
OUT_DIR   = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """\
You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply exactly "NOT FOUND".
Return only the answer, as a short value or phrase, not a full sentence."""

# ── Load labels ───────────────────────────────────────────────────────────────
labels_all: dict[str, dict] = {}
for split_file in [
    _ROOT / "data/financebench/sample/multi_cluster/random/sample_doc_labels.json",
    _ROOT / "data/financebench/sample/multi_cluster/random/unsampled_doc_labels.json",
]:
    if split_file.exists():
        labels_all.update(json.loads(split_file.read_text()))

selected = dict(sorted(labels_all.items())[:MAX_DOCS])
print(f"Selected {len(selected)} docs\n")

# ── Load gpt54 model for judge ────────────────────────────────────────────────
gpt54_mod = __import__("models.gpt54", fromlist=["client"])

# ── Run ───────────────────────────────────────────────────────────────────────
total = correct_count = no_hit_count = 0
records = []

for pdf_key, doc_labels in sorted(selected.items()):
    doc_name = pdf_key.replace(".pdf", "")
    doc_path = TEXT_DIR / f"{doc_name}.txt"
    if not doc_path.exists():
        print(f"SKIP (no txt): {doc_name}")
        continue

    ground_truth = doc_labels.get(QUESTION)
    document = runner_mod._load_txt_doc(doc_name, doc_path)
    doc_token_count = runner_mod._count_tokens(document["text"])

    t0 = time.time()
    apply_result = rule_apply_merge(
        document=document,
        rule_names=[RULE_NAME],
        question_slug=RULE_QUESTION_DIR,
        question=QUESTION,
        model_name=MODEL,
        rules_dir=str(RULES_DIR),
        output_dir=str(OUT_DIR / "_trace"),
        system_prompt=SYSTEM_PROMPT,
    )
    latency = time.time() - t0

    answer = apply_result.get("predicted_answer")
    correct = runner_mod._judge(DATASET, QUESTION, ground_truth, answer, gpt54_mod)

    total += 1
    if correct:
        correct_count += 1
    has_hit = bool(apply_result.get("rules_with_hits"))
    if not has_hit:
        no_hit_count += 1

    cost_ratio = round(apply_result.get("input_tokens", 0) / max(doc_token_count, 1), 4)
    record = {
        "doc_name": doc_name,
        "ground_truth": ground_truth,
        "answer": answer,
        "correct": correct,
        "status": "ok",
        "rules_with_hits": apply_result.get("rules_with_hits", []),
        "rules_with_no_hits": apply_result.get("rules_with_no_hits", []),
        "retrieved_token_count": apply_result.get("retrieved_token_count", 0),
        "input_tokens": apply_result.get("input_tokens", 0),
        "doc_token_count": doc_token_count,
        "cost_ratio": cost_ratio,
        "latency_seconds": round(latency, 2),
        "retrieved_text": apply_result.get("retrieved_text", ""),
    }
    records.append(record)
    (OUT_DIR / f"{doc_name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    marker = "✓" if correct else ("NO_HIT" if not has_hit else "✗")
    print(f"  {marker}  {doc_name:<55}  gt={str(ground_truth)[:35]!r}  pred={str(answer)[:35]!r}")

# ── Summary ───────────────────────────────────────────────────────────────────
accuracy    = correct_count / total if total else 0
no_hit_rate = no_hit_count  / total if total else 0

summary = {
    "n": total,
    "correct": correct_count,
    "accuracy": round(accuracy, 4),
    "no_hit": no_hit_count,
    "no_hit_rate": round(no_hit_rate, 4),
}
(OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(f"\nn={total}  accuracy={accuracy:.4f}  no_hit={no_hit_count} ({no_hit_rate:.1%})")
