"""Run rule_refine for every sample question using pre-computed sampled eval accuracy."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_refine import rule_refine

QUERIES_FILE   = "data/financebench/sample_queries.txt"
LABELS_FILE    = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_BASE_DIR = "rules/financebench_single_cluster/llm/gpt54/one_shot"
EVAL_DIR       = "results/financebench_single_cluster/llm/gpt54/one_shot/eval_merge"
OUTPUT_DIR     = "rules/financebench_single_cluster/llm/gpt54/refine"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for question in questions:
    slug          = make_slug(question)
    question_slug = f"{slug}_10"
    rule_folder   = Path(RULES_BASE_DIR) / f"{question_slug}_llm"
    refine_file   = Path(OUTPUT_DIR) / f"{question_slug}_refine.json"
    eval_file     = Path(EVAL_DIR) / f"{question_slug}_sampled.json"

    if refine_file.exists():
        print(f"SKIP (exists): {question_slug}")
        continue

    if not eval_file.exists():
        print(f"SKIP (no eval): {eval_file}")
        continue

    target_accuracy = json.loads(eval_file.read_text())["accuracy"]

    if not rule_folder.is_dir():
        print(f"SKIP (no rules): {rule_folder}")
        continue

    rule_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(rule_folder)
        if f.startswith("rule_") and f.endswith(".py")
    )
    if not rule_names:
        print(f"SKIP (empty): {rule_folder}")
        continue

    # Load ground truth and documents
    ground_truth = {k: labels[k][question] for k in labels if question in labels[k]}
    documents: list[dict] = []
    for pdf_key in labels:
        path = Path(PROCESSING_DIR) / f"{pdf_key.replace('.pdf', '')}_reconstructed.json"
        if path.exists():
            documents.append(json.loads(path.read_text(encoding="utf-8")))

    print(f"\nQuestion: {question}")
    print(f"  rules={len(rule_names)}  target_acc={target_accuracy:.2f}  docs={len(documents)}")

    try:
        result = rule_refine(
            rule_names=rule_names,
            target_accuracy=target_accuracy,
            question=question,
            question_slug=question_slug,
            documents=documents,
            ground_truth=ground_truth,
            rules_dir=RULES_BASE_DIR,
            output_dir=OUTPUT_DIR,
        )
        print(f"  selected={result['selected_rules_count']}/{result['all_rules_count']}  "
              f"acc={result['merge_accuracy']:.2f}  "
              f"cost_reduction={result['cost_reduction_ratio']:.2%}  "
              f"llm_calls={result['total_llm_calls']}")
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

# ── Summary table ──────────────────────────────────────────────────────────────
summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"\n{'=== Rule Refinement Summary ==='}")
    hdr = f"{'Question':<50}  {'All':>4}  {'Sel':>4}  {'Acc':>5}  {'CostSaved':>9}  {'LLMCalls':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(
            f"  {s['question'][:48]:<48}  {s['all_rules_count']:>4}  "
            f"{s['selected_rules_count']:>4}  {s['merge_accuracy']:>5.2f}  "
            f"{s['cost_reduction_ratio']:>8.1%}  {s['total_llm_calls']:>8}"
        )
    print(f"\nResults: {OUTPUT_DIR}/")
