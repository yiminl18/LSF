#!/usr/bin/env python3
"""Analyze documents to understand reporting period patterns."""

import json
import os
import re

# Document list
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

def search_answer_spans(doc, answer):
    """Find spans containing the answer."""
    answer_lower = answer.lower()
    matches = []
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "").lower()
        text_span = span.get("text_span", "").lower()
        if answer_lower in text or answer_lower in text_span:
            matches.append((i, span))
        # Also check for partial matches (date only)
        date_match = re.search(r'\w+\s+\d+,?\s+\d{4}', answer)
        if date_match:
            date_str = date_match.group().lower()
            if date_str in text or date_str in text_span:
                matches.append((i, span))
    return matches

def analyze_doc(doc_name, answer):
    print(f"\n{'='*60}")
    print(f"Doc: {doc_name}")
    print(f"Answer: {answer}")

    doc = load_doc(doc_name)
    matches = search_answer_spans(doc, answer)

    if matches:
        print(f"Found {len(matches)} matching spans:")
        seen = set()
        for idx, span in matches:
            if idx in seen:
                continue
            seen.add(idx)
            print(f"  Index {idx}:")
            print(f"    text: {span.get('text', '')[:100]}...")
            print(f"    text_span: {span.get('text_span', '')[:100]}...")
            print(f"    page: {span.get('page_no')}, label: {span.get('label')}")
            print(f"    path: {span.get('structure', {}).get('path_text', '')}")
            print(f"    level: {span.get('structure', {}).get('level')}, bold: {span.get('bold')}")
    else:
        print("No direct match found. Checking page 1 spans with period keywords...")
        # Look for spans with fiscal year, quarterly, date of report keywords
        for i, span in enumerate(doc.get("texts", [])[:30]):
            text = span.get("text", "").lower()
            text_span = span.get("text_span", "").lower()
            combined = text + " " + text_span
            keywords = ["fiscal year", "quarterly", "date of report", "period ended", "event reported"]
            if any(kw in combined for kw in keywords):
                print(f"  Keyword match at index {i}:")
                print(f"    text: {span.get('text', '')[:100]}")
                print(f"    text_span: {span.get('text_span', '')[:100]}")
                print(f"    page: {span.get('page_no')}, label: {span.get('label')}")
                print(f"    path: {span.get('structure', {}).get('path_text', '')}")

def main():
    gt = load_labels()
    print(f"Loaded {len(gt)} ground truth answers")

    for doc_name in DOCS:
        if doc_name in gt:
            analyze_doc(doc_name, gt[doc_name])

if __name__ == "__main__":
    main()
