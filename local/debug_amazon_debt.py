#!/usr/bin/env python3
"""Debug Amazon debt retrieval."""

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

# Check Amazon 2020
doc = load_doc("AMAZON_2020_10K")
spans = rule_long_term_debt_tables(doc)

print(f"Found {len(spans)} spans")
print("="*80)

for i, span in enumerate(spans):
    print(f"\nSpan {i}:")
    print(f"Page: {span.get('page_no')}")
    print(f"Path: {span.get('structure', {}).get('path_text', '')[:80]}")
    print("Text:")
    print(span.get('text', '')[:1500])
    print("-"*40)

# Search for 101,406 in the document
print("\n" + "="*80)
print("Searching for '101406' in the entire document...")
for i, span in enumerate(doc.get("texts", [])):
    text = span.get("text", "").replace(",", "")
    if "101406" in text:
        print(f"\nFound at index {i}, page {span.get('page_no')}, {span.get('label')}")
        print(f"Path: {span.get('structure', {}).get('path_text', '')[:80]}")
        print("Text:")
        print(span.get("text", "")[:500])
