#!/usr/bin/env python3
"""Analyze patterns for long-term debt spans."""

import json
import re
from pathlib import Path
from collections import defaultdict

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
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is long-term debt at year-end (0 if none)?"

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    result = {}
    for doc_name in DOCS:
        pdf_name = f"{doc_name}.pdf"
        if pdf_name in labels:
            result[doc_name] = labels[pdf_name].get(QUESTION)
    return result

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def find_debt_spans(doc):
    """Find spans related to debt."""
    results = []
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "").lower()
        path_text = span.get("structure", {}).get("path_text", "").lower()

        # Check for debt-related keywords
        if any(kw in text or kw in path_text for kw in [
            "long-term debt", "long term debt", "debt",
            "borrowing", "notes payable", "senior notes"
        ]):
            if span.get("label") == "table":
                results.append({
                    "index": i,
                    "page_no": span.get("page_no"),
                    "label": span.get("label"),
                    "level": span.get("structure", {}).get("level"),
                    "path_text": span.get("structure", {}).get("path_text", ""),
                    "has_long_term": "long-term" in text.lower() or "long term" in text.lower() or "long-term" in path_text.lower()
                })
    return results

def find_balance_sheet_tables(doc):
    """Find balance sheet tables."""
    results = []
    for i, span in enumerate(doc.get("texts", [])):
        path_text = span.get("structure", {}).get("path_text", "").lower()
        text = span.get("text", "").lower()

        if span.get("label") == "table":
            # Look for balance sheet patterns
            if any(kw in path_text or kw in text for kw in [
                "balance sheet", "financial position", "consolidated balance",
                "item 8", "item 6", "financial statements"
            ]):
                results.append({
                    "index": i,
                    "page_no": span.get("page_no"),
                    "path_text": span.get("structure", {}).get("path_text", ""),
                    "text_preview": span.get("text", "")[:150]
                })
    return results

def main():
    labels = load_labels()

    print("Analyzing debt-related table patterns...\n")

    # Collect path_text patterns for tables containing long-term debt
    path_patterns = defaultdict(int)

    for doc_name in DOCS:
        answer = labels.get(doc_name)
        doc = load_doc(doc_name)
        if not doc:
            continue

        print(f"\n{'='*60}")
        print(f"{doc_name} - Expected: {answer}")

        if answer == "0" or answer is None:
            print("  Skipping (no positive debt value)")
            continue

        # Find debt-related tables
        debt_spans = find_debt_spans(doc)
        balance_sheets = find_balance_sheet_tables(doc)

        print(f"  Found {len(debt_spans)} debt-related tables")
        for s in debt_spans[:3]:
            print(f"    Page {s['page_no']}: {s['path_text'][:80]}")
            path_parts = s['path_text'].split("|")
            for part in path_parts:
                path_patterns[part.strip().lower()] += 1

        print(f"  Found {len(balance_sheets)} balance sheet tables")
        for s in balance_sheets[:2]:
            print(f"    Page {s['page_no']}: {s['path_text'][:80]}")

    print("\n" + "="*60)
    print("Most common path_text patterns:")
    for pattern, count in sorted(path_patterns.items(), key=lambda x: -x[1])[:20]:
        if count >= 2:
            print(f"  {count}: {pattern}")

if __name__ == "__main__":
    main()
