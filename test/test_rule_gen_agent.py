"""Test rule_gen_agent on a sample query (default: first query in sample_queries.txt)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_gen_agent import rule_gen_agent

LABELS_FILE    = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
QUERIES_FILE   = "data/financebench/sample_queries.txt"

# Load question (default: first query; pass index as argv[1] to override)
queries = Path(QUERIES_FILE).read_text(encoding="utf-8").splitlines()
query_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
QUESTION = next(q.strip() for i, q in enumerate(queries) if q.strip() and i == query_idx)
print(f"Question: {QUESTION}\n")

# Load labels and documents
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
documents: list[dict] = []
for pdf_key in labels:
    stem = Path(pdf_key).stem
    doc_path = _ROOT / PROCESSING_DIR / f"{stem}_reconstructed.json"
    if doc_path.exists():
        documents.append(json.loads(doc_path.read_text(encoding="utf-8")))

print(f"Documents loaded: {len(documents)}")

# Build ground truth for this question
ground_truth: dict = {
    pdf_key: labels[pdf_key][QUESTION]
    for pdf_key in labels
    if QUESTION in labels[pdf_key]
}
print(f"Ground truth entries: {len(ground_truth)}")
print()

# Run agent
result = rule_gen_agent(documents, QUESTION, ground_truth)

# Print summary
print("\n=== Result Summary ===")
print(f"Agent turns     : {result['agent_turns']}")
print(f"Total LLM calls : {result['total_llm_calls']}")
print(f"Rules           : {len(result['rules'])}")
print(f"Merge accuracy  : {result['merge_accuracy']:.2f}  ({result['merge_num_correct']}/{result['num_documents']})")
print(f"Avg cost ratio  : {result['avg_cost_ratio']:.5f}")
print(f"Latency         : {result['latency_seconds']:.1f}s")
print(f"Agent in tokens : {result['agent_input_tokens']}")
print(f"Agent out tokens: {result['agent_output_tokens']}")
print(f"Judge in tokens : {result['judge_input_tokens']}")
print(f"Judge out tokens: {result['judge_output_tokens']}")
print()
print("Rules (name → hit_rate, avg_cost_ratio):")
for r in result["rules"]:
    print(f"  {r['rule_name']:<50}  hit_rate={r['hit_rate']:.2f}  avg_cost={r['avg_cost_ratio']:.5f}")
print()
print("Per-doc merge results:")
for doc_name, dr in result["merge_per_doc"].items():
    status = "✓" if dr["correct"] else "✗"
    print(
        f"  {status} {doc_name:<45}  "
        f"pred={str(dr.get('predicted',''))[:30]!r}  "
        f"gt={str(dr.get('ground_truth',''))!r}"
    )
print(f"\nRule dir : {result['rule_dir']}")
print(f"Log file : {result['log_file']}")
