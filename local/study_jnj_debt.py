"""Study Johnson & Johnson document for long-term debt."""

import json
import re
from pathlib import Path

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def main():
    doc = load_doc("JOHNSON_JOHNSON_2022_10K")
    texts = doc.get("texts", [])

    # Search for tables with debt-related content
    print("Searching for tables with 'debt' content...\n")

    debt_tables = []
    for i, span in enumerate(texts):
        if span.get("label") == "table":
            text = span.get("text", "").lower()
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "") for c in cells if c.get("is_row_header")]

            has_debt_keyword = any("debt" in h.lower() for h in row_headers) or "debt" in text

            if has_debt_keyword:
                debt_tables.append((i, span))

    print(f"Found {len(debt_tables)} tables with debt-related content\n")

    for idx, span in debt_tables:
        print(f"Table at index {idx}:")
        print(f"  page_no: {span.get('page_no')}")
        print(f"  path_text: {span.get('structure', {}).get('path_text', '')}")

        # Get row headers with debt
        cells = span.get("table_data", {}).get("cells", [])
        debt_headers = [c.get("text", "") for c in cells if c.get("is_row_header") and "debt" in c.get("text", "").lower()]
        print(f"  debt row headers: {debt_headers}")

        # Look for the value 26.9 or 26,900 or similar
        for cell in cells:
            cell_text = cell.get("text", "")
            if "26" in cell_text:
                print(f"    Found '26' in cell: {cell_text} at row {cell.get('row')}, col {cell.get('col')}")

        print(f"  text_preview: {span.get('text', '')[:300]}...")
        print()

    # Also search for "26.9" or "26900" anywhere in the document
    print("\n\nSearching for '26.9' or '26900' in all spans...")
    for i, span in enumerate(texts):
        text = span.get("text", "")
        if "26.9" in text or "26900" in text or "26,900" in text:
            print(f"\nSpan {i}: page {span.get('page_no')}, label {span.get('label')}")
            print(f"  path: {span.get('structure', {}).get('path_text', '')}")
            print(f"  text: {text[:400]}")

if __name__ == "__main__":
    main()
