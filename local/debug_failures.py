#!/usr/bin/env python3
"""Debug failing documents."""

import json
from pathlib import Path

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def rule_long_term_debt_tables(doc):
    return [s for s in doc.get("texts", [])
            if s.get("label") == "table" and
            ("long-term debt" in s.get("text", "").lower() or
             "long term debt" in s.get("text", "").lower())]

def rule_item6_selected_data(doc):
    return [s for s in doc.get("texts", [])
            if s.get("label") == "table" and
            ("item 6" in s.get("structure", {}).get("path_text", "").lower() or
             "selected financial" in s.get("structure", {}).get("path_text", "").lower()) and
            any(kw in s.get("text", "").lower() for kw in [
                "long-term", "long term", "total debt", "obligations"
            ])]

# Debug Amazon 2020
print("="*80)
print("AMAZON_2020_10K - Expected: $101,406 million")
doc = load_doc("AMAZON_2020_10K")

print("\nRule: rule_item6_selected_data")
spans = rule_item6_selected_data(doc)
print(f"Found {len(spans)} spans")
for span in spans:
    print(f"\nPage {span.get('page_no')}")
    print(f"Path: {span.get('structure', {}).get('path_text', '')[:80]}")
    print("Text:")
    print(span.get('text', '')[:800])

# Debug Costco 2018
print("\n" + "="*80)
print("COSTCO_2018_10K - Expected: 6487")
doc = load_doc("COSTCO_2018_10K")

# Search for 6487
print("\nSearching for '6487' in document...")
for i, span in enumerate(doc.get("texts", [])):
    if "6487" in span.get("text", "").replace(",", ""):
        print(f"\nIndex {i}, Page {span.get('page_no')}, {span.get('label')}")
        print(f"Path: {span.get('structure', {}).get('path_text', '')[:80]}")
        print("Text:")
        print(span.get('text', '')[:400])

# Debug 3M 2023Q2
print("\n" + "="*80)
print("3M_2023Q2_10Q - Expected: $12,954 million")
doc = load_doc("3M_2023Q2_10Q")

# Search for 12954
print("\nSearching for '12954' in document...")
for i, span in enumerate(doc.get("texts", [])):
    if "12954" in span.get("text", "").replace(",", ""):
        print(f"\nIndex {i}, Page {span.get('page_no')}, {span.get('label')}")
        print(f"Path: {span.get('structure', {}).get('path_text', '')[:80]}")
        print("Text:")
        print(span.get('text', '')[:600])
