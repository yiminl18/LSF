"""Test rule_gen_agent_coarse on the first sample query."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_gen_agent_coarse import rule_gen_agent_coarse

LABELS_FILE    = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
QUERIES_FILE   = "data/financebench/sample_queries.txt"

# Load question
QUESTION = Path(QUERIES_FILE).read_text(encoding="utf-8").splitlines()[0].strip()
print(f"Question: {QUESTION}\n")

# Load labels
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))

# Load documents
documents: list[dict] = []
for pdf_key in labels:
    stem = Path(pdf_key).stem
    doc_path = _ROOT / PROCESSING_DIR / f"{stem}_reconstructed.json"
    if doc_path.exists():
        documents.append(json.loads(doc_path.read_text(encoding="utf-8")))

print(f"Documents loaded: {len(documents)}")

# Build ground truth
ground_truth: dict = {
    pdf_key: labels[pdf_key][QUESTION]
    for pdf_key in labels
    if QUESTION in labels[pdf_key]
}
print(f"Ground truth entries: {len(ground_truth)}")
print()

# Run agent
result = rule_gen_agent_coarse(documents, QUESTION, ground_truth)

# Print summary
print("\n=== Result Summary ===")
print(f"Agent turns:      {result['agent_turns']}")
print(f"Rules generated:  {len(result['rules'])}")
best_indl = max((r["individual_accuracy"] for r in result["rules"]), default=0.0)
print(f"Best indl acc:    {best_indl:.2f}")
print(f"Merge accuracy:   {result['merge_accuracy']:.2f}  ({result['merge_num_correct']}/{result['num_documents']})")
print(f"Latency:          {result['latency_seconds']:.1f}s")
print(f"Input tokens:     {result['input_tokens']}")
print(f"Output tokens:    {result['output_tokens']}")
print()
print("Rules:")
for r in result["rules"]:
    print(f"  {r['rule_name']:<50}  indl={r['individual_accuracy']:.2f}  ({r['individual_num_correct']}/{result['num_documents']})")
