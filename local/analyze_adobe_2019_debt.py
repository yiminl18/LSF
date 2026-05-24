#!/usr/bin/env python3
"""Analyze Adobe 2019 10K debt tables."""

import json
from pathlib import Path

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

doc = load_doc("ADOBE_2019_10K")

print("Looking for debt tables and '988' pattern...")
print("="*60)

# Search for spans with the answer value
for i, span in enumerate(doc.get("texts", [])):
    text = span.get("text", "")
    path_text = span.get("structure", {}).get("path_text", "")

    # Check for the answer value
    if "988" in text.replace(",", "") or "long-term debt" in text.lower():
        print(f"\nIndex {i}, Page {span.get('page_no')}, {span.get('label')}")
        print(f"Path: {path_text[:80]}")
        print(f"Text: {text[:400]}")
        print("-"*40)
