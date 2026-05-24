#!/usr/bin/env python3
"""Analyze documents to find where long-term debt answers appear."""

import json
import re
from pathlib import Path

# The documents and ground truth
DOCS = [
    "BOEING_2019_10K",
    "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K",
    "AMCOR_2019_10K",
    "AMAZON_2020_10K",
    "AMAZON_2019_10K",
    "ADOBE_2021_10K",
    "EBAY_2022_10K",
    "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q",
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is long-term debt at year-end (0 if none)?"

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    result = {}
    for doc_name in DOCS:
        pdf_name = f"{doc_name}.pdf"
        if pdf_name in labels:
            result[doc_name] = labels[pdf_name].get(QUESTION)
    return result

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def extract_numeric_value(answer):
    """Extract numeric value from answer for matching."""
    if answer is None or answer == "0":
        return None
    # Extract numeric patterns
    text = str(answer).replace(",", "").replace("$", "")
    match = re.search(r'([\d.]+)', text)
    if match:
        return match.group(1)
    return None

def find_answer_spans(doc, answer):
    """Find spans containing the answer."""
    if answer is None:
        return []

    answer_str = str(answer).lower()
    numeric = extract_numeric_value(answer)

    matches = []
    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "").lower()
        # Check for exact match or numeric match
        if answer_str in text or (numeric and numeric in text.replace(",", "")):
            matches.append({
                "index": i,
                "page_no": span.get("page_no"),
                "label": span.get("label"),
                "level": span.get("structure", {}).get("level"),
                "path_text": span.get("structure", {}).get("path_text", "")[:100],
                "text_preview": span.get("text", "")[:200],
                "bold": span.get("bold")
            })
    return matches

def main():
    labels = load_labels()

    print("=" * 80)
    print(f"Analyzing: {QUESTION}")
    print("=" * 80)

    for doc_name in DOCS:
        print(f"\n{'='*60}")
        print(f"Document: {doc_name}")
        answer = labels.get(doc_name)
        print(f"Expected: {answer}")

        doc = load_doc(doc_name)
        if not doc:
            print("  ERROR: Document not found")
            continue

        total_spans = len(doc.get("texts", []))
        print(f"Total spans: {total_spans}")

        if answer == "0" or answer is None:
            # For 8K docs or null answers, look for any long-term debt keywords
            debt_spans = []
            for i, span in enumerate(doc.get("texts", [])):
                text = span.get("text", "").lower()
                if "long-term debt" in text or "long term debt" in text:
                    debt_spans.append({
                        "index": i,
                        "page_no": span.get("page_no"),
                        "label": span.get("label"),
                        "path_text": span.get("structure", {}).get("path_text", "")[:80],
                        "text_preview": span.get("text", "")[:200]
                    })
            if debt_spans:
                print(f"  Found {len(debt_spans)} spans with 'long-term debt':")
                for m in debt_spans[:5]:
                    print(f"    Page {m['page_no']}, {m['label']}, path: {m['path_text']}")
                    print(f"      Text: {m['text_preview'][:100]}")
            else:
                print("  No spans with 'long-term debt' keyword")
            continue

        matches = find_answer_spans(doc, answer)

        if matches:
            print(f"  Found {len(matches)} matching spans:")
            for m in matches[:10]:
                print(f"    Index {m['index']}: Page {m['page_no']}, {m['label']}, {m['level']}")
                print(f"      Path: {m['path_text']}")
                print(f"      Text: {m['text_preview'][:100]}")
        else:
            print("  NO MATCHES FOUND - searching for debt keywords")
            for i, span in enumerate(doc.get("texts", [])):
                text = span.get("text", "").lower()
                if "long-term debt" in text or "long term debt" in text:
                    print(f"    Index {i}: Page {span.get('page_no')}, {span.get('label')}")
                    print(f"      {span.get('text', '')[:200]}")

if __name__ == "__main__":
    main()
