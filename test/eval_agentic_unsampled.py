"""Evaluate Pareto-selected rule sets on 50 unsampled docs with gpt54.

Reads selected_rules per question from selected_rules_agent/<slug>.json,
runs `rule_apply_merge` + gpt54 judge on each unsampled doc, writes per-question
JSON + a summary.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_apply_merge import rule_apply_merge

# ── Config ────────────────────────────────────────────────────────────────────
QUERIES_FILE    = "data/financebench/sample_queries.txt"
LABELS_FILE     = "data/financebench/sample/single_cluster/random/unsampled_doc_labels.json"
PROCESSING_DIR  = "data/financebench/processing"
RULES_BASE_DIR  = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot"
SELECTED_DIR    = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selected_rules_agent"
MERGE_RUN_DIR   = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_agentic/run_unsampled"
OUTPUT_DIR      = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_agentic"
EVAL_MODEL_NAME = "gpt54"

model_mod = importlib.import_module(f"models.{EVAL_MODEL_NAME}")

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
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def judge(question: str, ground_truth, predicted) -> bool:
    if ground_truth is None or predicted is None:
        return False
    gt_str   = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted)
    user_msg = f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {pred_str}"
    try:
        response = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
    except Exception as e:
        if "content_filter" in str(e) or "content management" in str(e):
            warnings.warn(f"Judge content filter, treating as incorrect")
            return False
        raise
    verdict = (response.choices[0].message.content or "").strip().lower()
    return verdict == "correct"


# ── Load questions, labels, docs ─────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
print(f"Questions: {len(questions)}  |  Unsampled docs: {len(labels)}\n")

doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))

doc_total_tokens: dict[str, int] = {
    doc_name: count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
    for doc_name, doc in doc_map.items()
}

Path(MERGE_RUN_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

summary: list[dict] = []

for question in questions:
    slug          = make_slug(question)
    rule_slug     = f"{slug}_10_llm"
    output_slug   = f"{slug}_10"

    selected_path = Path(SELECTED_DIR) / f"{rule_slug}.json"
    out_file      = Path(OUTPUT_DIR) / f"{output_slug}_unsampled.json"

    if not selected_path.exists():
        print(f"SKIP (no selection): {selected_path}")
        continue
    if out_file.exists():
        print(f"SKIP (eval exists): {out_file.name}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append({
            "question":           question,
            "question_slug":      rule_slug,
            "rules_count":        len(existing.get("selected_rules", [])),
            "unsampled_accuracy": existing.get("accuracy"),
            "avg_cost_ratio":     existing.get("avg_cost_ratio"),
        })
        continue

    sel_data = json.loads(selected_path.read_text(encoding="utf-8"))
    rule_names = sel_data.get("selected_rules", [])
    if not rule_names:
        print(f"SKIP (empty selection): {rule_slug}")
        continue

    print(f"\nQuestion: {question}")
    print(f"  selected rules: {len(rule_names)}  | unsampled docs: {len(doc_map)}")

    per_doc: list[dict] = []
    cost_ratios: list[float] = []

    for doc_name, document in doc_map.items():
        try:
            result = rule_apply_merge(
                document=document,
                rule_names=rule_names,
                question_slug=rule_slug,
                question=question,
                model_name=EVAL_MODEL_NAME,
                rules_dir=RULES_BASE_DIR,
                output_dir=MERGE_RUN_DIR,
            )
            predicted = result.get("predicted_answer")
            retrieved_tokens = result.get("retrieved_token_count", 0)
        except Exception as e:
            print(f"  ERROR rule_apply on {doc_name}: {e}")
            predicted = None
            retrieved_tokens = 0

        gt = labels.get(doc_name + ".pdf", {}).get(question)
        try:
            correct = judge(question, gt, predicted)
        except Exception as e:
            print(f"  ERROR judge on {doc_name}: {e}")
            correct = False

        total_tok = doc_total_tokens.get(doc_name, 1)
        cost_ratio = retrieved_tokens / total_tok if total_tok > 0 else 0.0
        cost_ratios.append(cost_ratio)

        per_doc.append({
            "doc_name":         doc_name,
            "predicted":        predicted,
            "ground_truth":     gt,
            "correct":          correct,
            "retrieved_tokens": retrieved_tokens,
            "total_doc_tokens": total_tok,
            "cost_ratio":       round(cost_ratio, 6),
        })

    accuracy = sum(1 for r in per_doc if r["correct"]) / max(1, len(per_doc))
    avg_cost = mean(cost_ratios) if cost_ratios else 0.0

    out = {
        "question":          question,
        "question_slug":     rule_slug,
        "eval_model":        EVAL_MODEL_NAME,
        "selection_source":  str(selected_path),
        "selected_rules":    rule_names,
        "accuracy":          round(accuracy, 4),
        "avg_cost_ratio":    round(avg_cost, 6),
        "num_documents":     len(per_doc),
        "per_doc":           per_doc,
    }
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → unsampled_acc={accuracy:.2f}  avg_cost={avg_cost:.5f}  saved={out_file.name}")

    summary.append({
        "question":           question,
        "question_slug":      rule_slug,
        "rules_count":        len(rule_names),
        "unsampled_accuracy": round(accuracy, 4),
        "avg_cost_ratio":     round(avg_cost, 6),
    })

# ── Cross-question summary ───────────────────────────────────────────────────
print(f"\n{'Question':<55}  {'#Rules':>6}  {'uAcc':>5}  {'avg_cost':>9}")
print("-" * 82)
for s in summary:
    print(f"  {s['question'][:53]:<53}  {s['rules_count']:>6}  "
          f"{s['unsampled_accuracy']:>5.2f}  {s['avg_cost_ratio']:>9.5f}")

summary_path = Path(OUTPUT_DIR) / "summary_unsampled.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nUnsampled summary: {summary_path}")
