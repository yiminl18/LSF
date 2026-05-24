#!/usr/bin/env python3
"""Debug Johnson & Johnson total assets retrieval."""
import json
from pathlib import Path

doc_name = "JOHNSON_JOHNSON_2022_10K"
path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
with open(path) as f:
    doc = json.load(f)

def rule_table_total_assets_balance_sheet(doc: dict) -> list[dict]:
    """Match first 2 tables with total assets row header in balance sheet/financial sections."""
    try:
        results = []
        path_keywords = ["item 6", "item 8", "balance sheet", "selected financial",
                        "financial statement", "consolidated balance", "annual report", "part iv"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if h.strip() in ["total assets", "total assets (i)"]:
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []

spans = rule_table_total_assets_balance_sheet(doc)
print(f"Retrieved {len(spans)} spans")
for i, span in enumerate(spans):
    print(f"\nSpan {i+1}:")
    print(f"  Page: {span.get('page_no')}")
    print(f"  Path: {span.get('structure', {}).get('path_text', '')[:80]}")
    print(f"  Text preview: {span.get('text', '')[:300]}")

    # Look at row with total assets
    cells = span.get("table_data", {}).get("cells", [])
    for c in cells:
        if c.get("is_row_header") and "total assets" in c.get("text", "").lower():
            row = c.get("row")
            print(f"  Total assets row ({row}):")
            row_cells = [cc for cc in cells if cc.get("row") == row]
            for rc in row_cells:
                print(f"    col {rc.get('col')}: {rc.get('text')}")
