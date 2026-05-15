"""Select minimal-cost rule subsets on unsampled docs using gpt-5.4-mini + rule_refine.py.

For each question:
  - candidate pool : rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm/
  - training docs  : all 50 unsampled docs
  - target accuracy: all-rules merge accuracy on unsampled (from one_shot eval_merge)
  - model          : gpt54mini (gpt-5.4-mini)

Output: rules/financebench_single_cluster/llm/gpt54mini/refine_unsampled/
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_refine import rule_refine

QUERIES_FILE        = "data/financebench/sample_queries.txt"
LABELS_FILE         = "data/financebench/unsampled_doc_labels.json"
PROCESSING_DIR      = "data/financebench/processing"
ONE_SHOT_RULES_DIR  = "rules/financebench_single_cluster/llm/gpt54/one_shot"
OUTPUT_DIR          = "rules/financebench_single_cluster/llm/gpt54mini/refine_unsampled"
MODEL_NAME          = "gpt54mini"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))

doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")

documents = list(doc_map.values())
print(f"Questions: {len(questions)}  |  Unsampled docs: {len(documents)}\n")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

total_start = time.time()

for question in questions:
    slug     = make_slug(question)
    out_slug = f"{slug}_10"

    result_path = Path(OUTPUT_DIR) / f"{out_slug}_refine.json"
    if result_path.exists():
        print(f"SKIP (exists): {result_path.name}")
        continue

    rule_folder = Path(ONE_SHOT_RULES_DIR) / f"{out_slug}_llm"
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

    print(f"\n{'='*70}")
    print(f"Question : {question}")
    print(f"Rules    : {len(rule_names)}  |  Docs: {len(documents)}")

    try:
        result = rule_refine(
            rule_names=rule_names,
            question=question,
            question_slug=out_slug,
            documents=documents,
            ground_truth=labels,
            rules_dir=ONE_SHOT_RULES_DIR,
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME,
            target_accuracy=None,  # auto-computed using gpt54mini
        )
        print(
            f"  selected={result['selected_rules_count']}  "
            f"accuracy={result['merge_accuracy']:.4f}  "
            f"cost_ratio={result['avg_cost_ratio']:.5f}  "
            f"llm_calls={result['total_llm_calls']}  "
            f"latency={result['total_latency_seconds']:.1f}s  "
            f"input_tok={result['total_input_tokens']}  "
            f"output_tok={result['total_output_tokens']}"
        )
    except Exception as e:
        import traceback
        print(f"  FATAL ERROR: {e}")
        traceback.print_exc()
        continue

total_wall = round(time.time() - total_start, 1)
print(f"\n\nTotal wall time: {total_wall}s")

summary_path = Path(OUTPUT_DIR) / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"\n{'Question':<55}  {'Sel':>3}  {'Acc':>5}  {'CostR':>8}  {'Calls':>6}  {'Lat(s)':>7}")
    print("-" * 95)
    for s in summary:
        print(
            f"  {s['question'][:53]:<53}  "
            f"{s['selected_rules_count']:>3}  "
            f"{s['merge_accuracy']:>5.2f}  "
            f"{s['avg_cost_ratio']:>8.5f}  "
            f"{s['total_llm_calls']:>6}  "
            f"{s['total_latency_seconds']:>7.1f}"
        )
