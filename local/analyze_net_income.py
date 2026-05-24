#!/usr/bin/env python3
"""Analyze net income span locations across sampled documents."""

import json
import os
import re

DOCS = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
]

QUESTION = "What is net income (loss) for the most recent fiscal year?"

# Load labels
with open("data/financebench/sample_doc_labels.json") as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in DOCS:
    key = f"{doc_name}.pdf"
    if key in labels and QUESTION in labels[key]:
        ground_truth[doc_name] = labels[key][QUESTION]

print("Ground truth values:")
for doc, val in ground_truth.items():
    print(f"  {doc}: {val}")
print()

# Analyze each document
for doc_name in DOCS:
    doc_path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        print(f"Document not found: {doc_path}")
        continue

    with open(doc_path) as f:
        doc = json.load(f)

    gt_value = ground_truth.get(doc_name, "")
    # Extract numeric part for searching
    numeric_match = re.search(r"[\d,\.]+", gt_value)
    numeric_val = numeric_match.group() if numeric_match else gt_value

    print(f"\n{'='*60}")
    print(f"Document: {doc_name}")
    print(f"Ground truth: {gt_value} (searching for: {numeric_val})")
    print(f"Total spans: {len(doc.get('texts', []))}")

    found_spans = []
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "")
        if numeric_val in text:
            found_spans.append((i, span))

    print(f"Spans containing '{numeric_val}': {len(found_spans)}")

    for idx, span in found_spans[:5]:  # Limit output
        path_text = (span.get("structure") or {}).get("path_text", "N/A")
        print(f"\n  Span {idx}:")
        print(f"    Label: {span.get('label')}")
        print(f"    Page: {span.get('page_no')}")
        print(f"    Level: {(span.get('structure') or {}).get('level', 'N/A')}")
        print(f"    Path: {path_text[:100]}...")
        text_preview = span.get("text", "")[:200].replace("\n", " ")
        print(f"    Text: {text_preview}...")
