#!/usr/bin/env python3
"""Analyze page 1 patterns for reporting period."""

import json
import re

DOCS = [
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

QUESTION = "What is the reporting period covered by this document (e.g. fiscal year ended, quarter ended, or event date)?"

def load_doc(doc_name):
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    gt = {}
    for doc_name in DOCS:
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            gt[doc_name] = labels[key][QUESTION]
    return gt

def analyze_page1_patterns(doc_name, answer):
    """Find page 1 spans with period keywords."""
    doc = load_doc(doc_name)

    # Detect document type
    doc_type = "10-K" if "10K" in doc_name else ("10-Q" if "10Q" in doc_name else "8-K")

    print(f"\n{'='*70}")
    print(f"Doc: {doc_name} ({doc_type})")
    print(f"Answer: {answer}")

    # Keywords for each type
    period_keywords = [
        "fiscal year ended",
        "quarterly period ended",
        "date of report",
        "date of earliest event",
        "for the fiscal year",
        "for the quarterly period",
    ]

    # Check page 1 and 2 spans
    matches = []
    for i, span in enumerate(doc.get("texts", [])):
        page = span.get("page_no", 0)
        if page > 2:
            continue

        text = span.get("text", "").lower()
        text_span = span.get("text_span", "").lower()
        combined = text + " " + text_span

        for kw in period_keywords:
            if kw in combined:
                matches.append((i, span, kw))
                break

    if matches:
        for idx, span, kw in matches:
            print(f"  Match ('{kw}') at index {idx}:")
            print(f"    text: {span.get('text', '')[:120]}")
            if span.get('text_span'):
                print(f"    text_span: {span.get('text_span', '')[:120]}")
            print(f"    page: {span.get('page_no')}, label: {span.get('label')}, bold: {span.get('bold')}")
            print(f"    path: {span.get('structure', {}).get('path_text', '')[:80]}")
            print(f"    level: {span.get('structure', {}).get('level')}")

            # Check if answer is in this span
            answer_lower = answer.lower()
            combined = span.get("text", "").lower() + " " + span.get("text_span", "").lower()
            if answer_lower in combined:
                print(f"    *** CONTAINS FULL ANSWER ***")
            else:
                # Check partial date match
                date_match = re.search(r'(\w+\s+\d+,?\s+\d{4})', answer)
                if date_match and date_match.group(1).lower() in combined:
                    print(f"    *** CONTAINS DATE: {date_match.group(1)} ***")
    else:
        print("  No period keyword matches on pages 1-2")
        # Fallback: check first 20 spans
        print("  First 10 spans:")
        for i, span in enumerate(doc.get("texts", [])[:10]):
            print(f"    {i}: page={span.get('page_no')}, {span.get('text', '')[:80]}...")

def main():
    gt = load_labels()
    print(f"Loaded {len(gt)} ground truth answers")

    for doc_name in DOCS:
        if doc_name in gt:
            analyze_page1_patterns(doc_name, gt[doc_name])

if __name__ == "__main__":
    main()
