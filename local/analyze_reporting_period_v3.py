#!/usr/bin/env python3
"""Analyze page 1 patterns for reporting period."""

import json
import os
import re

# Map from expected doc names to actual filenames
DOC_MAP = {
    "BOEING_2019_10K": "BOEING_2019_10K",
    "ADOBE_2020_10K": "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K": "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K": "COSTCO_2018_10K",
    "AMCOR_2019_10K": "AMCOR_2019_10K",
    "AMAZON_2020_10K": "AMAZON_2020_10K",
    "AMAZON_2019_10K": "AMAZON_2019_10K",
    "ADOBE_2021_10K": "ADOBE_2021_10K",
    "EBAY_2022_10K": "EBAY_2022_10K",
    "ADOBE_2019_10K": "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q": "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q": "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q": None,  # Not available
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q": "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01": "AMCOR_2022_8K_dated-2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09": "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16": "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01": None,  # Not available
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "FOOTLOCKER_2022_8K_dated-2022-05-20",
}

QUESTION = "What is the reporting period covered by this document (e.g. fiscal year ended, quarter ended, or event date)?"

def load_doc(doc_name):
    # Check mapping
    actual_name = DOC_MAP.get(doc_name, doc_name)
    if actual_name is None:
        return None
    path = f"data/financebench/processing/{actual_name}_reconstructed.json"
    if not os.path.exists(path):
        print(f"  WARNING: File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    gt = {}
    for doc_name in DOC_MAP.keys():
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            gt[doc_name] = labels[key][QUESTION]
    return gt

def analyze_page1_patterns(doc_name, answer):
    """Find page 1 spans with period keywords."""
    doc = load_doc(doc_name)
    if doc is None:
        print(f"\n{'='*70}")
        print(f"Doc: {doc_name} - SKIPPED (file not found)")
        return None

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

    result = {"doc_name": doc_name, "answer": answer, "doc_type": doc_type, "matches": []}

    if matches:
        for idx, span, kw in matches:
            answer_lower = answer.lower()
            combined = span.get("text", "").lower() + " " + span.get("text_span", "").lower()
            contains_answer = answer_lower in combined

            # Check partial date match
            date_match = re.search(r'(\w+\s+\d+,?\s+\d{4})', answer)
            contains_date = date_match and date_match.group(1).lower() in combined

            match_info = {
                "index": idx,
                "keyword": kw,
                "text": span.get("text", "")[:120],
                "text_span": span.get("text_span", "")[:120],
                "page": span.get("page_no"),
                "label": span.get("label"),
                "bold": span.get("bold"),
                "path": span.get("structure", {}).get("path_text", ""),
                "level": span.get("structure", {}).get("level"),
                "contains_answer": contains_answer,
                "contains_date": contains_date,
            }
            result["matches"].append(match_info)

            print(f"  Match ('{kw}') at index {idx}:")
            print(f"    text: {match_info['text']}")
            if match_info['text_span']:
                print(f"    text_span: {match_info['text_span']}")
            print(f"    page: {match_info['page']}, label: {match_info['label']}, bold: {match_info['bold']}")
            print(f"    path: {match_info['path'][:80]}")
            print(f"    level: {match_info['level']}")
            if contains_answer:
                print(f"    *** CONTAINS FULL ANSWER ***")
            elif contains_date:
                print(f"    *** CONTAINS DATE ***")
    else:
        print("  No period keyword matches on pages 1-2")
        # Fallback: check first 20 spans
        print("  First 10 spans:")
        for i, span in enumerate(doc.get("texts", [])[:10]):
            print(f"    {i}: page={span.get('page_no')}, {span.get('text', '')[:80]}...")

    return result

def main():
    gt = load_labels()
    print(f"Loaded {len(gt)} ground truth answers")

    results = []
    for doc_name in DOC_MAP.keys():
        if doc_name in gt:
            result = analyze_page1_patterns(doc_name, gt[doc_name])
            if result:
                results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total docs analyzed: {len(results)}")

    has_match = sum(1 for r in results if r["matches"])
    has_answer = sum(1 for r in results if any(m["contains_answer"] or m["contains_date"] for m in r["matches"]))
    print(f"Docs with period keyword matches: {has_match}")
    print(f"Docs where matches contain answer/date: {has_answer}")

if __name__ == "__main__":
    main()
