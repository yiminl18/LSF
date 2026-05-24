"""Driver: apply agentic-selected rules with full-pool fallback on multi-cluster unsampled docs.

Reads selected rule sets from selected_rules_agent/<slug>.json.
For each unsampled doc, calls default_rule.apply_with_fallback (gpt54mini gate
+ gpt54 QA), then judges the prediction with gpt54.

Writes per-question JSON + summary to eval_agentic_fallback/.
"""

from __future__ import annotations

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

from default_rule               import apply_with_fallback
from rule_refinement.eval_judge import judge

# ── Config ────────────────────────────────────────────────────────────────────
QUERIES_FILE    = "data/financebench/mix_doc_queries.txt"
LABELS_FILE     = "data/financebench/sample/multi_cluster/random/unsampled_doc_labels.json"
PROCESSING_DIR  = "data/financebench/processing"
RULES_BASE_DIR  = "rules/financebench/lsf/multi_clusters/llm/gpt54/one_shot"
SELECTED_DIR    = "results/financebench/lsf/multi_clusters/llm/gpt54/one_shot/selected_rules_agent"
OUTPUT_DIR      = "results/financebench/lsf/multi_clusters/llm/gpt54/one_shot/eval_agentic_fallback"
RELEVANCE_MODEL = "gpt54mini"
QA_MODEL        = "gpt54"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


# ── Load questions, labels, docs ─────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
print(f"Questions: {len(questions)}  |  Unsampled docs in labels: {len(labels)}", flush=True)

doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
print(f"Docs loaded: {len(doc_map)}\n", flush=True)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
summary: list[dict] = []

# ── Per-question processing ──────────────────────────────────────────────────
for question in questions:
    slug        = make_slug(question)
    rule_slug   = f"{slug}_18_llm"
    output_slug = f"{slug}_18"

    selected_path = Path(SELECTED_DIR) / f"{rule_slug}.json"
    out_file      = Path(OUTPUT_DIR) / f"{output_slug}_unsampled.json"

    if not selected_path.exists():
        print(f"SKIP (no selection): {rule_slug}", flush=True)
        continue
    if out_file.exists():
        print(f"SKIP (eval exists): {out_file.name}", flush=True)
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append({
            "question":              question,
            "question_slug":         rule_slug,
            "accuracy":              existing.get("accuracy"),
            "fallback_rate":         existing.get("fallback_rate"),
            "refined_rules_count":   existing.get("refined_rules_count"),
            "full_pool_count":       existing.get("full_pool_count"),
        })
        continue

    sel_data      = json.loads(selected_path.read_text(encoding="utf-8"))
    refined_rules = sel_data.get("selected_rules", [])
    if not refined_rules:
        print(f"SKIP (empty refined rules): {rule_slug}", flush=True)
        continue

    rule_folder = Path(RULES_BASE_DIR) / rule_slug
    all_rules = sorted([
        f.stem for f in rule_folder.glob("rule_*.py")
        if f.is_file()
    ])
    if not all_rules:
        print(f"SKIP (no rule folder): {rule_folder}", flush=True)
        continue

    print(f"\n{'='*70}", flush=True)
    print(f"Question: {question}", flush=True)
    print(f"  refined: {len(refined_rules)} rules  |  full pool: {len(all_rules)} rules", flush=True)

    per_doc:        list[dict] = []
    total_mini_in   = 0
    total_mini_out  = 0
    total_gpt54_in  = 0
    total_gpt54_out = 0
    judge_in        = 0
    judge_out       = 0
    fallback_count  = 0
    correct_count   = 0

    for doc_name, document in doc_map.items():
        gt = labels.get(doc_name + ".pdf", {}).get(question)
        try:
            result = apply_with_fallback(
                document=document,
                question=question,
                refined_rules=refined_rules,
                all_rules=all_rules,
                rule_folder=rule_folder,
                relevance_model=RELEVANCE_MODEL,
                qa_model=QA_MODEL,
            )
        except Exception as e:
            print(f"  ERR apply on {doc_name}: {e}", flush=True)
            continue

        try:
            correct, j_in, j_out = judge(question, gt, result["predicted"], model_name=QA_MODEL)
            judge_in  += j_in
            judge_out += j_out
        except Exception as e:
            print(f"  ERR judge on {doc_name}: {e}", flush=True)
            correct = False

        if correct:
            correct_count += 1
        if result["fallback_triggered"]:
            fallback_count += 1

        toks = result["tokens"]
        total_mini_in   += toks["mini_in"]
        total_mini_out  += toks["mini_out"]
        total_gpt54_in  += toks["gpt54_in"]
        total_gpt54_out += toks["gpt54_out"]

        per_doc.append({
            "doc_name":                  doc_name,
            "predicted":                 result["predicted"],
            "ground_truth":              gt,
            "correct":                   correct,
            "fallback_triggered":        result["fallback_triggered"],
            "relevance_verdict":         result["relevance_verdict"],
            "retrieved_tokens_refined":  result["retrieved_tokens_refined"],
            "retrieved_tokens_full":     result["retrieved_tokens_full"],
            "retrieved_tokens_used":     result["retrieved_tokens_used"],
            "tokens":                    toks,
        })
        flag = "Y" if correct else "N"
        fb   = "F" if result["fallback_triggered"] else "."
        pred = str(result["predicted"])[:55] if result["predicted"] is not None else "None"
        print(f"  [{flag}{fb}] {doc_name}: {pred}", flush=True)

    n = len(per_doc)
    accuracy      = correct_count / n if n else 0.0
    fallback_rate = fallback_count / n if n else 0.0

    out = {
        "question":             question,
        "question_slug":        rule_slug,
        "refined_rules_count":  len(refined_rules),
        "full_pool_count":      len(all_rules),
        "num_documents":        n,
        "accuracy":             round(accuracy, 4),
        "fallback_rate":        round(fallback_rate, 4),
        "models":               {"relevance": RELEVANCE_MODEL, "qa": QA_MODEL},
        "total_tokens": {
            "gpt54mini_in":  total_mini_in,
            "gpt54mini_out": total_mini_out,
            "gpt54_in":      total_gpt54_in + judge_in,
            "gpt54_out":     total_gpt54_out + judge_out,
        },
        "per_doc": per_doc,
    }
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → uAcc={accuracy:.2f}  fallback_rate={fallback_rate:.2f}  saved={out_file.name}", flush=True)

    summary.append({
        "question":              question,
        "question_slug":         rule_slug,
        "accuracy":              round(accuracy, 4),
        "fallback_rate":         round(fallback_rate, 4),
        "refined_rules_count":   len(refined_rules),
        "full_pool_count":       len(all_rules),
        "total_tokens": {
            "gpt54mini_in":  total_mini_in,
            "gpt54mini_out": total_mini_out,
            "gpt54_in":      total_gpt54_in + judge_in,
            "gpt54_out":     total_gpt54_out + judge_out,
        },
    })

# ── Cross-question summary ───────────────────────────────────────────────────
print(f"\n{'Question':<55}  {'uAcc':>5}  {'fb_rate':>8}  {'mini_in':>9}  {'gpt54_in':>9}", flush=True)
print("-" * 100, flush=True)
for s in summary:
    tt = s.get("total_tokens", {})
    print(f"  {s['question'][:53]:<53}  "
          f"{s['accuracy']:>5.2f}  "
          f"{s['fallback_rate']:>8.2f}  "
          f"{tt.get('gpt54mini_in', 0):>9,}  "
          f"{tt.get('gpt54_in', 0):>9,}",
          flush=True)

print(f"\nMean uAcc: {mean(s['accuracy'] for s in summary):.3f}", flush=True)
print(f"Mean fallback rate: {mean(s['fallback_rate'] for s in summary):.3f}", flush=True)

summary_path = Path(OUTPUT_DIR) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSummary: {summary_path}", flush=True)
print(f"Results: {OUTPUT_DIR}/", flush=True)
