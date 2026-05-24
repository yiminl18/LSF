#!/usr/bin/env python3
"""Debug J&J document for net income."""

import json
from pathlib import Path

doc_path = Path("/home/yiminglin/LSF/data/financebench/processing/JOHNSON_JOHNSON_2022_10K_reconstructed.json")

with open(doc_path) as f:
    doc = json.load(f)

print("Ground truth: $17.9 billion")
print("\n=== Spans containing '17.9' ===")
for i, span in enumerate(doc.get("texts", [])):
    text = span.get("text", "")
    if "17.9" in text or "17,9" in text:
        print(f"\nIndex {i}:")
        print(f"  page_no: {span.get('page_no')}")
        print(f"  label: {span.get('label')}")
        print(f"  path_text: {span.get('structure', {}).get('path_text', '')[:100]}")
        print(f"  text preview: {text[:300]}...")

print("\n\n=== Tables in Item 8 with net income headers ===")
for i, span in enumerate(doc.get("texts", [])):
    if span.get("label") != "table":
        continue
    path_text = span.get("structure", {}).get("path_text", "").lower()
    if "item 8" not in path_text and "financial statement" not in path_text:
        continue
    cells = span.get("table_data", {}).get("cells", [])
    row_headers = [c.get("text", "") for c in cells if c.get("is_row_header")]
    for h in row_headers:
        if "net income" in h.lower() or "net earnings" in h.lower():
            print(f"\nIndex {i}: page {span.get('page_no')}")
            print(f"  path_text: {path_text[:100]}")
            print(f"  matching header: {h}")
            print(f"  text preview: {span.get('text', '')[:400]}...")
            break

print("\n\n=== All tables with 'net earnings' row header ===")
for i, span in enumerate(doc.get("texts", [])):
    if span.get("label") != "table":
        continue
    cells = span.get("table_data", {}).get("cells", [])
    row_headers = [c.get("text", "") for c in cells if c.get("is_row_header")]
    for h in row_headers:
        if "net earnings" in h.lower():
            print(f"\nIndex {i}: page {span.get('page_no')}")
            print(f"  path_text: {span.get('structure', {}).get('path_text', '')[:100]}")
            print(f"  matching header: {h}")
            # Check if 17.9 is in this table
            has_179 = "17.9" in span.get("text", "") or "17,9" in span.get("text", "")
            print(f"  contains 17.9: {has_179}")
            print(f"  text: {span.get('text', '')[:500]}...")
            break
