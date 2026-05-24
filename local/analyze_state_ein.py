#!/usr/bin/env python3
"""Analyze documents to find state/EIN patterns and test rules."""

import json
import os
import re
import time
from pathlib import Path

START_TIME = time.time()

# Sampled documents
SAMPLED_DOCS = [
    "BOEING_2019_10K",
    "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K",
    "AMCOR_2019_10K",
    "AMAZON_2020_10K",
    "AMAZON_2019_10K",
    "ADOBE_2021_10K",
    "EBAY_2022_10K",
    "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q",
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is the state (or other jurisdiction) of incorporation and the IRS Employer Identification Number?"

# Load ground truth
with open("data/financebench/sample_mix_doc_labels.json") as f:
    all_labels = json.load(f)

ground_truth = {}
for doc_name in SAMPLED_DOCS:
    pdf_name = doc_name + ".pdf"
    if pdf_name in all_labels and QUESTION in all_labels[pdf_name]:
        ground_truth[doc_name] = all_labels[pdf_name][QUESTION]

print(f"Ground truth for {len(ground_truth)} docs:")
for k, v in ground_truth.items():
    print(f"  {k}: {v}")

# Load documents
docs = {}
for doc_name in SAMPLED_DOCS:
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if os.path.exists(path):
        with open(path) as f:
            docs[doc_name] = json.load(f)
    else:
        print(f"Missing: {path}")

print(f"\nLoaded {len(docs)} documents")

# Analyze patterns - look for spans containing state/EIN info
def analyze_doc(doc_name, doc, gt_answer):
    """Find spans matching the ground truth."""
    texts = doc.get("texts", [])
    gt_lower = gt_answer.lower()

    # Parse ground truth - usually format like "Delaware, 91-0425694"
    parts = re.split(r'[,;]\s*', gt_answer)
    state_part = parts[0].strip() if parts else ""
    ein_part = parts[1].strip() if len(parts) > 1 else ""

    found_state = []
    found_ein = []
    found_jurisdiction_label = []
    found_ein_label = []

    for i, span in enumerate(texts):
        text = span.get("text", "")
        text_lower = text.lower()
        page = span.get("page_no", 0)

        # Look for state
        if state_part.lower() in text_lower:
            found_state.append({
                "idx": i,
                "page": page,
                "text": text[:100],
                "label": span.get("label"),
                "bold": span.get("bold"),
                "level": span.get("structure", {}).get("level"),
                "path": span.get("structure", {}).get("path_text", "")
            })

        # Look for EIN
        if ein_part and ein_part in text:
            found_ein.append({
                "idx": i,
                "page": page,
                "text": text[:100],
                "label": span.get("label"),
                "bold": span.get("bold"),
                "level": span.get("structure", {}).get("level"),
                "path": span.get("structure", {}).get("path_text", "")
            })

        # Look for jurisdiction label
        if "jurisdiction" in text_lower and "incorporation" in text_lower:
            found_jurisdiction_label.append({
                "idx": i,
                "page": page,
                "text": text[:100],
                "label": span.get("label"),
            })

        # Look for EIN label
        if "employer identification" in text_lower or "i.r.s." in text_lower:
            found_ein_label.append({
                "idx": i,
                "page": page,
                "text": text[:100],
                "label": span.get("label"),
            })

    return {
        "state": state_part,
        "ein": ein_part,
        "found_state": found_state,
        "found_ein": found_ein,
        "found_jurisdiction_label": found_jurisdiction_label,
        "found_ein_label": found_ein_label,
    }

print("\n=== PATTERN ANALYSIS ===")
for doc_name in SAMPLED_DOCS:
    if doc_name not in docs or doc_name not in ground_truth:
        print(f"\n{doc_name}: MISSING")
        continue

    result = analyze_doc(doc_name, docs[doc_name], ground_truth[doc_name])
    print(f"\n{doc_name}:")
    print(f"  GT: {ground_truth[doc_name]}")
    print(f"  State matches on pages: {[s['page'] for s in result['found_state']]}")
    print(f"  EIN matches on pages: {[s['page'] for s in result['found_ein']]}")
    print(f"  Jurisdiction label pages: {[s['page'] for s in result['found_jurisdiction_label']]}")
    print(f"  EIN label pages: {[s['page'] for s in result['found_ein_label']]}")

    # Show first state match
    if result['found_state']:
        s = result['found_state'][0]
        print(f"  First state match: page={s['page']}, label={s['label']}, bold={s['bold']}, level={s['level']}")
        print(f"    Text: {s['text'][:60]}")

    # Show first EIN match
    if result['found_ein']:
        s = result['found_ein'][0]
        print(f"  First EIN match: page={s['page']}, label={s['label']}, bold={s['bold']}, level={s['level']}")
        print(f"    Text: {s['text'][:60]}")
