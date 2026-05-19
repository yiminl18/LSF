"""Run rule_gen_llm_coarse for every question in sample_queries.txt."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_gen_llm_coarse import rule_gen_llm_coarse

QUERIES_FILE   = "data/financebench/sample_queries.txt"
LABELS_FILE    = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR = "data/financebench/processing"
RULES_DIR      = "rules/financebench_single_cluster/llm/gpt54/one_shot"
OUTPUT_DIR     = "results/financebench_single_cluster/llm/gpt54/one_shot/rule_gen"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


# Load questions
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
print(f"Questions: {len(questions)}")

# Load labels and documents
labels: dict[str, dict] = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
documents: list[dict] = []
for pdf_key in labels:
    doc_path = Path(PROCESSING_DIR) / f"{pdf_key.replace('.pdf', '')}_reconstructed.json"
    if doc_path.exists():
        documents.append(json.loads(doc_path.read_text(encoding="utf-8")))
    else:
        print(f"WARNING: missing {doc_path}")

print(f"Documents loaded: {len(documents)}\n")

# Ensure output dirs exist
Path(RULES_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

processed = 0
skipped = 0

for question in questions:
    slug = make_slug(question)
    rule_folder = Path(RULES_DIR) / f"{slug}_{len(documents)}_llm"

    if rule_folder.is_dir():
        print(f"SKIP (exists): {rule_folder}")
        skipped += 1
        continue

    print(f"\nGenerating: {question}")

    ground_truth = {
        pdf_key: labels[pdf_key][question]
        for pdf_key in labels
        if question in labels[pdf_key]
    }

    try:
        result = rule_gen_llm_coarse(
            documents=documents,
            question=question,
            ground_truth=ground_truth,
            output_dir=OUTPUT_DIR,
            rules_dir=RULES_DIR,
        )
        print(f"  rules: {len(result['rules'])}  latency: {result['latency_seconds']:.1f}s")
        processed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        continue

print(f"\n=== Done ===")
print(f"Processed: {processed} / {len(questions)} questions")
print(f"Skipped:   {skipped} (already had rules)")
print(f"Rules dir: {RULES_DIR}/")
print(f"Results:   {OUTPUT_DIR}/")
