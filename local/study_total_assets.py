#!/usr/bin/env python3
"""Study total assets location across sampled documents."""
import json
import re
import time

START_TIME = time.time()

QUESTION = "What is total assets at year-end (from the audited balance sheet)?"
DOC_NAMES = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
]

# Load ground truth
with open("data/financebench/sample_doc_labels.json") as f:
    all_labels = json.load(f)

ground_truth = {}
for doc_name in DOC_NAMES:
    key = f"{doc_name}.pdf"
    if key in all_labels and QUESTION in all_labels[key]:
        ground_truth[doc_name] = all_labels[key][QUESTION]

print("Ground truth values:")
for doc_name, answer in ground_truth.items():
    print(f"  {doc_name}: {answer}")
print()

# Load documents
docs = {}
for doc_name in DOC_NAMES:
    with open(f"data/financebench/processing/{doc_name}_reconstructed.json") as f:
        docs[doc_name] = json.load(f)

# Study where answers appear
print("="*80)
print("SEARCHING FOR ANSWER SPANS")
print("="*80)

for doc_name, doc in docs.items():
    answer = ground_truth.get(doc_name, "")
    # Extract numeric portion for matching
    numeric_match = re.search(r'[\d,\.]+', answer.replace(',', ''))
    if numeric_match:
        answer_num = numeric_match.group().replace(',', '')
    else:
        answer_num = answer

    print(f"\n{doc_name}: Answer = {answer}")
    found_spans = []

    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "")
        # Check if answer value is in the text
        if answer_num and answer_num in text.replace(',', '').replace(' ', ''):
            found_spans.append({
                "idx": i,
                "page_no": span.get("page_no"),
                "label": span.get("label"),
                "level": span.get("structure", {}).get("level"),
                "path_text": span.get("structure", {}).get("path_text", "")[:100],
                "text_preview": text[:150].replace('\n', ' '),
            })

    if found_spans:
        print(f"  Found in {len(found_spans)} spans:")
        for s in found_spans[:5]:
            print(f"    idx={s['idx']}, page={s['page_no']}, label={s['label']}, level={s['level']}")
            print(f"    path_text: {s['path_text']}")
            print(f"    text: {s['text_preview']}")
    else:
        print(f"  NOT FOUND via numeric match")
        # Try text search
        for i, span in enumerate(doc.get("texts", [])):
            text = span.get("text", "").lower()
            if "total assets" in text:
                print(f"    idx={i}, page={span.get('page_no')}, label={span.get('label')}")
                print(f"    path_text: {span.get('structure', {}).get('path_text', '')[:100]}")
                print(f"    text: {text[:200]}")

print("\n" + "="*80)
print("STUDYING TABLE PATTERNS")
print("="*80)

for doc_name, doc in docs.items():
    print(f"\n{doc_name}:")
    for i, span in enumerate(doc.get("texts", [])):
        if span.get("label") != "table":
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]

        # Check for total assets row header
        for h in row_headers:
            if "total assets" in h:
                path_text = span.get("structure", {}).get("path_text", "")
                print(f"  TABLE idx={i}, page={span.get('page_no')}")
                print(f"    path_text: {path_text[:100]}")
                print(f"    row_header: {h[:50]}")
                # Show cell values in that row
                row_idx = None
                for c in cells:
                    if c.get("is_row_header") and "total assets" in c.get("text", "").lower():
                        row_idx = c.get("row")
                        break
                if row_idx is not None:
                    row_cells = [c for c in cells if c.get("row") == row_idx]
                    values = [c.get("text", "") for c in row_cells]
                    print(f"    row values: {values[:5]}")
                break
