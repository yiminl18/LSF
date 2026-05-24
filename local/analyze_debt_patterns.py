"""Analyze patterns for long-term debt across all documents."""

import json
import re
from pathlib import Path

GROUND_TRUTH = {
    "AMCOR_2019_10K": "5,314.4",  # million
    "COSTCO_2017_10K": "6,573",
    "BOEING_2018_10K": "10,657",
    "AMAZON_2018_10K": "50,708",  # 2018 10K has "Long-term debt" on balance sheet
    "EBAY_2021_10K": "7,727",
    "AMAZON_2016_10K": "7,694",
    "CORNING_2022_10K": "6,687",
    "NIKE_2021_10K": "9,413",
    "LOCKHEEDMARTIN_2022_10K": "15,547",
    "JOHNSON_JOHNSON_2022_10K": "26,888",  # in millions, not 26.9 billion
}

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def normalize_number(s):
    """Normalize number for comparison."""
    return s.replace(",", "").replace("$", "").strip()

def find_best_debt_tables(doc):
    """Find tables most likely to contain long-term debt."""
    results = []
    texts = doc.get("texts", [])

    for i, span in enumerate(texts):
        if span.get("label") != "table":
            continue

        cells = span.get("table_data", {}).get("cells", [])
        path_text = span.get("structure", {}).get("path_text", "").lower()

        # Check if any row header contains "long-term debt" or "long term debt"
        row_headers = [c.get("text", "") for c in cells if c.get("is_row_header")]
        has_lt_debt_header = any("long-term debt" in h.lower() or "long term debt" in h.lower() for h in row_headers)

        # Check path for relevant keywords
        relevant_path = any(kw in path_text for kw in ["balance sheet", "item 8", "borrowing", "debt"])

        if has_lt_debt_header or (relevant_path and "debt" in path_text):
            results.append({
                "index": i,
                "page_no": span.get("page_no"),
                "path_text": path_text,
                "has_lt_debt_header": has_lt_debt_header,
                "row_headers": [h for h in row_headers if "debt" in h.lower()],
            })

    return results

def check_value_in_table(span, value):
    """Check if value is in the table."""
    text = span.get("text", "")
    cells = span.get("table_data", {}).get("cells", [])

    norm_value = normalize_number(value)

    # Check in full text
    if norm_value in text.replace(",", ""):
        return True

    # Check in cells
    for cell in cells:
        if norm_value in cell.get("text", "").replace(",", ""):
            return True

    return False

def main():
    print("Pattern Analysis for Long-Term Debt Rules\n")
    print("=" * 80)

    all_patterns = []

    for doc_name, expected_value in GROUND_TRUTH.items():
        print(f"\n{doc_name}")
        print(f"  Expected value: {expected_value}")

        doc = load_doc(doc_name)
        tables = find_best_debt_tables(doc)

        print(f"  Found {len(tables)} candidate tables")

        for t in tables:
            span = doc["texts"][t["index"]]
            has_value = check_value_in_table(span, expected_value)

            if has_value:
                print(f"  ✓ Table at index {t['index']}, page {t['page_no']}")
                print(f"    path: {t['path_text'][:80]}...")
                print(f"    debt headers: {t['row_headers']}")
                all_patterns.append({
                    "doc": doc_name,
                    "page": t["page_no"],
                    "path": t["path_text"],
                    "has_lt_debt_header": t["has_lt_debt_header"],
                })

    print("\n" + "=" * 80)
    print("\nSUMMARY OF PATTERNS:\n")

    # Analyze path patterns
    balance_sheet_count = sum(1 for p in all_patterns if "balance sheet" in p["path"])
    item8_count = sum(1 for p in all_patterns if "item 8" in p["path"])
    borrowing_count = sum(1 for p in all_patterns if "borrowing" in p["path"])
    debt_note_count = sum(1 for p in all_patterns if "debt" in p["path"] and ("note" in p["path"] or "." in p["path"].split("debt")[0][-20:]))

    print(f"Tables with 'balance sheet' in path: {balance_sheet_count}")
    print(f"Tables with 'item 8' in path: {item8_count}")
    print(f"Tables with 'borrowing' in path: {borrowing_count}")
    print(f"Tables with debt note section: {debt_note_count}")
    print(f"Tables with 'long-term debt' row header: {sum(1 for p in all_patterns if p['has_lt_debt_header'])}")

    print("\n\nRECOMMENDED RULES:")
    print("1. Rule: Tables with 'long-term debt' row header")
    print("2. Rule: Tables in 'balance sheet' sections")
    print("3. Rule: Tables in debt/borrowing note sections")

if __name__ == "__main__":
    main()
