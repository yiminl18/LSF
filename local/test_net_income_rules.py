#!/usr/bin/env python3
"""Test net income rules against sampled documents."""

import json
import os
import re
import tiktoken

DOCS = [
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

QUESTION = "What is net income (loss) for the most recent fiscal year?"

# Load labels
with open("data/financebench/sample_doc_labels.json") as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in DOCS:
    key = f"{doc_name}.pdf"
    if key in labels and QUESTION in labels[key]:
        ground_truth[doc_name] = labels[key][QUESTION]

# Load all documents
documents = {}
for doc_name in DOCS:
    doc_path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    with open(doc_path) as f:
        documents[doc_name] = json.load(f)

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def get_doc_tokens(doc):
    """Get total tokens in document."""
    full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
    return len(enc.encode(full_text))

def get_span_tokens(spans):
    """Get tokens in spans."""
    text = "\n".join(s.get("text", "") for s in spans)
    return len(enc.encode(text))

def extract_numeric(val):
    """Extract numeric pattern for substring matching."""
    match = re.search(r"[\d,\.]+", val.replace("$", ""))
    return match.group() if match else val

# Rule definitions
def rule_income_statement_table(doc: dict) -> list[dict]:
    """Tables under Consolidated Statement of Income/Operations section."""
    try:
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            for seg in segs:
                if any(kw in seg for kw in [
                    "statement of income",
                    "statements of income",
                    "statement of operations",
                    "statements of operations",
                    "statement of comprehensive income",
                    "statements of comprehensive income",
                ]):
                    out.append(sp)
                    break
        return out
    except Exception:
        return []

def rule_selected_financial_data_table(doc: dict) -> list[dict]:
    """Tables under Selected Financial Data section."""
    try:
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            for seg in segs:
                if "selected" in seg and ("financial" in seg or "consolidated" in seg):
                    out.append(sp)
                    break
        return out
    except Exception:
        return []

def rule_net_income_row_table(doc: dict) -> list[dict]:
    """Tables containing a row with 'net income' or 'net earnings' text."""
    try:
        import re
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            # Check table text for net income keywords
            text = sp.get("text", "").lower()
            if re.search(r'\bnet\s+(income|earnings|loss)\b', text):
                # Check it's in a row header position (starts a row or is a cell label)
                cells = (sp.get("table_data") or {}).get("cells", [])
                for cell in cells:
                    if cell.get("is_row_header") and re.search(r'\bnet\s+(income|earnings|loss)\b', cell.get("text", "").lower()):
                        out.append(sp)
                        break
                else:
                    # Fallback: check if pattern appears at start of a line in markdown
                    lines = text.split('\n')
                    for line in lines:
                        line_clean = line.strip().lstrip('|').strip()
                        if re.match(r'^net\s+(income|earnings|loss)', line_clean):
                            out.append(sp)
                            break
        return out
    except Exception:
        return []

def rule_item8_table(doc: dict) -> list[dict]:
    """Tables under Item 8 (Financial Statements)."""
    try:
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            for seg in segs:
                if "item 8" in seg or seg.startswith("8.") or "financial statements" in seg:
                    out.append(sp)
                    break
        return out
    except Exception:
        return []

# Test each rule
rules = [
    ("rule_income_statement_table", rule_income_statement_table),
    ("rule_selected_financial_data_table", rule_selected_financial_data_table),
    ("rule_net_income_row_table", rule_net_income_row_table),
    ("rule_item8_table", rule_item8_table),
]

print("=" * 80)
print("Individual Rule Performance")
print("=" * 80)

for rule_name, rule_func in rules:
    hits = 0
    total_cost = 0
    for doc_name, doc in documents.items():
        gt_val = ground_truth.get(doc_name, "")
        numeric_val = extract_numeric(gt_val)

        spans = rule_func(doc)
        retrieved_text = "\n".join(s.get("text", "") for s in spans)

        hit = numeric_val in retrieved_text
        if hit:
            hits += 1

        doc_tokens = get_doc_tokens(doc)
        span_tokens = get_span_tokens(spans)
        cost = span_tokens / doc_tokens if doc_tokens > 0 else 0
        total_cost += cost

    avg_cost = total_cost / len(documents)
    print(f"\n{rule_name}:")
    print(f"  Hit rate: {hits}/{len(documents)} ({100*hits/len(documents):.0f}%)")
    print(f"  Avg cost: {avg_cost:.4f} ({100*avg_cost:.2f}%)")

# Test union of all rules
print("\n" + "=" * 80)
print("Union of All Rules")
print("=" * 80)

union_hits = 0
union_total_cost = 0
uncovered_docs = []

for doc_name, doc in documents.items():
    gt_val = ground_truth.get(doc_name, "")
    numeric_val = extract_numeric(gt_val)

    # Collect all spans from all rules, deduplicated by identity (id)
    all_spans = []
    seen_ids = set()
    for rule_name, rule_func in rules:
        for sp in rule_func(doc):
            sp_id = id(sp)
            if sp_id not in seen_ids:
                seen_ids.add(sp_id)
                all_spans.append(sp)

    retrieved_text = "\n".join(s.get("text", "") for s in all_spans)

    hit = numeric_val in retrieved_text
    if hit:
        union_hits += 1
    else:
        uncovered_docs.append(doc_name)

    doc_tokens = get_doc_tokens(doc)
    span_tokens = get_span_tokens(all_spans)
    cost = span_tokens / doc_tokens if doc_tokens > 0 else 0
    union_total_cost += cost

    print(f"{doc_name}: hit={hit}, cost={cost:.4f}, spans={len(all_spans)}")

union_avg_cost = union_total_cost / len(documents)
print(f"\nUnion hit rate: {union_hits}/{len(documents)} ({100*union_hits/len(documents):.0f}%)")
print(f"Union avg cost: {union_avg_cost:.4f} ({100*union_avg_cost:.2f}%)")

if uncovered_docs:
    print(f"\nUncovered documents: {uncovered_docs}")
