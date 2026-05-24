#!/usr/bin/env python3
"""Analyze document patterns for address of principal executive offices and ZIP code."""

import json
import os
import re
import time

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

QUESTION = "What is the address of principal executive offices and ZIP code?"

# Load labels
with open("data/financebench/sample_mix_doc_labels.json") as f:
    labels = json.load(f)

# Build ground truth dict
ground_truth = {}
for doc_name in SAMPLED_DOCS:
    pdf_name = doc_name + ".pdf"
    if pdf_name in labels and QUESTION in labels[pdf_name]:
        ground_truth[doc_name] = labels[pdf_name][QUESTION]

print(f"Ground truth for {len(ground_truth)} documents:")
for doc_name, answer in ground_truth.items():
    print(f"  {doc_name}: {answer}")
print()

# Analyze each document
for doc_name in SAMPLED_DOCS:
    doc_path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        print(f"MISSING: {doc_name}")
        continue

    with open(doc_path) as f:
        doc = json.load(f)

    answer = ground_truth.get(doc_name, "")
    if not answer:
        print(f"NO ANSWER: {doc_name}")
        continue

    print(f"\n{'='*80}")
    print(f"DOC: {doc_name}")
    print(f"ANSWER: {answer}")
    print(f"{'='*80}")

    # Find spans containing any part of the answer
    answer_lower = answer.lower()
    # Extract key components from answer
    # Try to find zip code pattern
    zip_match = re.search(r'\d{5}(?:-\d{4})?', answer)
    zip_code = zip_match.group() if zip_match else ""

    # Extract street address (usually first part before city)
    address_parts = answer.split(",")
    street_part = address_parts[0].strip() if address_parts else ""

    print(f"ZIP: {zip_code}")
    print(f"STREET: {street_part}")

    # Find matching spans
    matching_spans = []
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "")
        text_lower = text.lower()

        # Check for exact answer or key parts
        has_zip = zip_code and zip_code in text
        has_street = street_part.lower()[:20] in text_lower if street_part else False
        has_principal = "principal executive" in text_lower
        has_address_label = "address of" in text_lower and "principal" in text_lower
        has_zip_label = "(zip code)" in text_lower or "zip code" in text_lower

        if has_zip or has_street or has_principal or has_address_label or has_zip_label:
            matching_spans.append({
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "page_no": span.get("page_no"),
                "label": span.get("label"),
                "level": span.get("structure", {}).get("level"),
                "path_text": span.get("structure", {}).get("path_text", "")[:60],
                "bold": span.get("bold"),
                "has_zip": has_zip,
                "has_street": has_street,
                "has_address_label": has_address_label,
            })

    # Print matching spans
    for s in matching_spans[:15]:  # Limit output
        print(f"  [{s['index']}] page={s['page_no']} label={s['label']} level={s['level']} bold={s['bold']}")
        print(f"       has_zip={s['has_zip']} has_street={s['has_street']} has_addr_label={s['has_address_label']}")
        print(f"       text: {s['text']}")
        print(f"       path: {s['path_text']}")
