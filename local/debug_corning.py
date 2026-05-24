#!/usr/bin/env python3
"""Debug CORNING_2022_10K path_text patterns."""
import json

with open("data/financebench/processing/CORNING_2022_10K_reconstructed.json") as f:
    doc = json.load(f)

print("Tables with 'total assets' row header in CORNING_2022_10K:")
for i, span in enumerate(doc.get("texts", [])):
    if span.get("label") != "table":
        continue
    cells = span.get("table_data", {}).get("cells", [])
    row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
    for h in row_headers:
        if "total assets" in h:
            path_text = span.get("structure", {}).get("path_text", "")
            print(f"  idx={i}, page={span.get('page_no')}")
            print(f"  path_text: {path_text}")
            print(f"  row_header: {h}")
            break

print()
print("Looking for balance sheet path patterns:")
for i, span in enumerate(doc.get("texts", [])):
    path_text = span.get("structure", {}).get("path_text", "").lower()
    if "balance" in path_text or "financial" in path_text:
        print(f"  idx={i}, page={span.get('page_no')}, label={span.get('label')}")
        print(f"  path_text: {span.get('structure', {}).get('path_text', '')[:100]}")
