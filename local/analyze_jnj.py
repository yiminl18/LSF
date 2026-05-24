#!/usr/bin/env python3
"""Analyze Johnson & Johnson document for net income."""

import json

doc_path = "data/financebench/processing/JOHNSON_JOHNSON_2022_10K_reconstructed.json"
with open(doc_path) as f:
    doc = json.load(f)

# Search for "17.9" in all spans
print("Spans containing '17.9':")
for i, sp in enumerate(doc.get("texts", [])):
    text = sp.get("text", "")
    if "17.9" in text or "17,9" in text:
        path_text = (sp.get("structure") or {}).get("path_text", "N/A")
        print(f"\n  Span {i}:")
        print(f"    Label: {sp.get('label')}")
        print(f"    Page: {sp.get('page_no')}")
        print(f"    Level: {(sp.get('structure') or {}).get('level', 'N/A')}")
        print(f"    Path: {path_text}")
        text_preview = text[:300].replace("\n", " ")
        print(f"    Text: {text_preview}...")

# Also search for "net income" in tables to see the format
print("\n\nTables containing 'net income' or 'net earnings':")
import re
for i, sp in enumerate(doc.get("texts", [])):
    if sp.get("label") != "table":
        continue
    text = sp.get("text", "").lower()
    if re.search(r'\bnet\s+(income|earnings)\b', text):
        path_text = (sp.get("structure") or {}).get("path_text", "N/A")
        print(f"\n  Span {i}:")
        print(f"    Page: {sp.get('page_no')}")
        print(f"    Path: {path_text}")
        # Check for table_data cells
        cells = (sp.get("table_data") or {}).get("cells", [])
        if cells:
            header_cells = [c for c in cells if c.get("is_row_header") or c.get("is_column_header")]
            print(f"    Header cells ({len(header_cells)}):", [c.get("text")[:30] for c in header_cells[:10]])
