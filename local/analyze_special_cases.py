#!/usr/bin/env python3
"""Analyze special cases: eBay (null) and 8K documents (0)."""

import json
from pathlib import Path

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# Check eBay 10K - answer is null
print("="*60)
print("EBAY_2022_10K - Answer: null")
doc = load_doc("EBAY_2022_10K")
if doc:
    # Find tables with debt info
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "").lower()
        path_text = span.get("structure", {}).get("path_text", "").lower()
        if span.get("label") == "table" and ("long-term debt" in text or "long-term debt" in path_text):
            print(f"\nPage {span.get('page_no')}, path: {span.get('structure', {}).get('path_text', '')[:80]}")
            print(f"Text preview: {span.get('text', '')[:400]}")

# Check an 8K document
print("\n" + "="*60)
print("AMCOR_2022_8K_2022-07-01 - Answer: 0")
doc = load_doc("AMCOR_2022_8K_2022-07-01")
if doc:
    print(f"Total spans: {len(doc.get('texts', []))}")
    # Show all spans
    for i, span in enumerate(doc.get("texts", [])[:30]):
        print(f"\n{i}: Page {span.get('page_no')}, {span.get('label')}")
        print(f"   Path: {span.get('structure', {}).get('path_text', '')[:60]}")
        print(f"   Text: {span.get('text', '')[:100]}")

print("\n" + "="*60)
print("COSTCO_2023_8K_dated-2023-08-09 - Answer: 0")
doc = load_doc("COSTCO_2023_8K_dated-2023-08-09")
if doc:
    print(f"Total spans: {len(doc.get('texts', []))}")
    # Look for financial data
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "").lower()
        if any(kw in text for kw in ["debt", "financial", "balance", "liability", "borrow"]):
            print(f"\nPage {span.get('page_no')}, {span.get('label')}")
            print(f"   Path: {span.get('structure', {}).get('path_text', '')[:60]}")
            print(f"   Text: {span.get('text', '')[:150]}")
