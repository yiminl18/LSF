#!/usr/bin/env python3
"""Analyze net income patterns in sampled documents."""

import json
import re
from pathlib import Path

SAMPLED_DOCS = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K"
]

QUESTION = "What is net income (loss) for the most recent fiscal year?"
PROCESSING_DIR = Path("/home/yiminglin/LSF/data/financebench/processing")
LABELS_FILE = Path("/home/yiminglin/LSF/data/financebench/sample_doc_labels.json")

def load_documents():
    """Load all sampled documents."""
    docs = {}
    for doc_name in SAMPLED_DOCS:
        path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        if path.exists():
            with open(path) as f:
                docs[doc_name] = json.load(f)
        else:
            print(f"WARNING: {path} not found")
    return docs

def load_ground_truth():
    """Load ground truth answers."""
    with open(LABELS_FILE) as f:
        labels = json.load(f)

    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            ground_truth[doc_name] = labels[key][QUESTION]
    return ground_truth

def extract_numeric(text):
    """Extract numeric value from answer text."""
    # Remove $ and common text
    text = text.replace("$", "").replace(",", "")
    # Find numbers
    match = re.search(r'[\d,]+\.?\d*', text)
    if match:
        return match.group().replace(",", "")
    return None

def find_answer_spans(doc, answer):
    """Find spans containing the answer."""
    matching_spans = []
    answer_lower = answer.lower()
    answer_numeric = extract_numeric(answer)

    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "")
        text_lower = text.lower()

        # Check for exact match or numeric match
        match = False
        if answer_lower in text_lower:
            match = True
        elif answer_numeric:
            if answer_numeric in text.replace(",", ""):
                match = True

        if match:
            matching_spans.append({
                "index": i,
                "span": span,
                "page_no": span.get("page_no"),
                "label": span.get("label"),
                "level": span.get("structure", {}).get("level"),
                "path_text": span.get("structure", {}).get("path_text", ""),
                "bold": span.get("bold"),
                "size": span.get("size"),
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            })

    return matching_spans

def analyze_tables_with_net_income(doc):
    """Find tables that contain net income in row headers."""
    tables = []
    for i, span in enumerate(doc.get("texts", [])):
        if span.get("label") != "table":
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
        col_headers = [c.get("text", "").lower() for c in cells if c.get("is_column_header")]

        # Check for net income variations in row headers
        for h in row_headers:
            if "net income" in h or "net loss" in h or "net earnings" in h:
                tables.append({
                    "index": i,
                    "page_no": span.get("page_no"),
                    "path_text": span.get("structure", {}).get("path_text", ""),
                    "matching_row_header": h,
                    "col_headers": col_headers[:5],  # First 5 column headers
                    "text_preview": span.get("text", "")[:500]
                })
                break
    return tables

def main():
    docs = load_documents()
    ground_truth = load_ground_truth()

    print("=" * 80)
    print("NET INCOME ANALYSIS")
    print("=" * 80)

    for doc_name in SAMPLED_DOCS:
        print(f"\n{'=' * 80}")
        print(f"DOCUMENT: {doc_name}")
        print(f"{'=' * 80}")

        if doc_name not in docs:
            print("Document not found!")
            continue

        doc = docs[doc_name]
        answer = ground_truth.get(doc_name, "UNKNOWN")
        print(f"Ground truth answer: {answer}")

        # Find spans containing the answer
        print(f"\n--- Spans containing answer ---")
        matching = find_answer_spans(doc, answer)
        if matching:
            for m in matching[:5]:  # Show first 5
                print(f"  Index {m['index']}: page={m['page_no']}, label={m['label']}, level={m['level']}")
                print(f"    path_text: {m['path_text'][:100]}...")
                print(f"    text: {m['text_preview'][:150]}...")
        else:
            print("  No matching spans found!")

        # Find tables with net income headers
        print(f"\n--- Tables with net income row headers ---")
        tables = analyze_tables_with_net_income(doc)
        if tables:
            for t in tables[:3]:  # Show first 3
                print(f"  Index {t['index']}: page={t['page_no']}")
                print(f"    path_text: {t['path_text'][:100]}...")
                print(f"    matching_header: {t['matching_row_header']}")
                print(f"    col_headers: {t['col_headers']}")
        else:
            print("  No tables with net income headers found!")

if __name__ == "__main__":
    main()
