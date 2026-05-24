#!/usr/bin/env python3
"""Test tighter rules to reduce cost."""

import json
import re
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text):
        return int(len(text.split()) * 1.3)

SAMPLED_DOCS = [
    "AMCOR_2019_10K", "COSTCO_2017_10K", "BOEING_2018_10K", "AMAZON_2018_10K",
    "EBAY_2021_10K", "AMAZON_2016_10K", "CORNING_2022_10K", "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K", "JOHNSON_JOHNSON_2022_10K"
]

QUESTION = "What is net income (loss) for the most recent fiscal year?"
PROCESSING_DIR = Path("/home/yiminglin/LSF/data/financebench/processing")
LABELS_FILE = Path("/home/yiminglin/LSF/data/financebench/sample_doc_labels.json")

def load_documents():
    docs = {}
    for doc_name in SAMPLED_DOCS:
        path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        if path.exists():
            with open(path) as f:
                docs[doc_name] = json.load(f)
    return docs

def load_ground_truth():
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            ground_truth[doc_name] = labels[key][QUESTION]
    return ground_truth

def extract_numeric_value(text):
    text = text.lower().replace(",", "").replace(" ", "").replace("$", "")
    match = re.search(r'([\d.]+)\s*(billion|million|b|m)?', text)
    if match:
        num = float(match.group(1))
        unit = match.group(2) or ""
        if "billion" in unit or unit == "b":
            return num * 1000
        return num
    return None

def check_hit(retrieved_text, answer):
    retrieved_clean = retrieved_text.lower().replace(",", "").replace(" ", "")
    answer_clean = answer.lower().replace(",", "").replace(" ", "")
    if answer_clean in retrieved_clean:
        return True
    answer_nums = re.findall(r'[\d,]+\.?\d*', answer)
    for num in answer_nums:
        if num.replace(",", "") in retrieved_text.replace(",", ""):
            return True
    if "billion" in answer.lower():
        answer_value = extract_numeric_value(answer)
        if answer_value:
            target_low = int(answer_value * 0.99)
            target_high = int(answer_value * 1.01)
            text_nums = re.findall(r'[\d,]+', retrieved_text)
            for tn in text_nums:
                try:
                    if target_low <= int(tn.replace(",", "")) <= target_high:
                        return True
                except ValueError:
                    continue
    return False

def compute_cost(retrieved_spans, doc):
    full_text = " ".join(span.get("text", "") for span in doc.get("texts", []))
    retrieved_text = " ".join(span.get("text", "") for span in retrieved_spans)
    full_tokens = count_tokens(full_text)
    retrieved_tokens = count_tokens(retrieved_text)
    return retrieved_tokens / full_tokens if full_tokens else 0.0

# Original rule
def rule_original(doc):
    results = []
    financial_keywords = ["item 6", "item 7", "item 8", "selected financial",
                        "financial statement", "management's discussion"]
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path_text = span.get("structure", {}).get("path_text", "").lower()
        if not any(kw in path_text for kw in financial_keywords):
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
        for h in row_headers:
            if ("net income" in h or "net earnings" in h or "net loss" in h) and "per share" not in h:
                results.append(span)
                break
    return results

# Tighter rule - Item 8 only
def rule_item8_only(doc):
    results = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path_text = span.get("structure", {}).get("path_text", "").lower()
        if "item 8" not in path_text and "financial statement" not in path_text:
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
        for h in row_headers:
            if ("net income" in h or "net earnings" in h) and "per share" not in h:
                results.append(span)
                break
    return results

# Tighter rule - Item 8 + Item 6
def rule_item8_item6(doc):
    results = []
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path_text = span.get("structure", {}).get("path_text", "").lower()
        if "item 8" not in path_text and "item 6" not in path_text and "financial statement" not in path_text and "selected financial" not in path_text:
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
        for h in row_headers:
            if ("net income" in h or "net earnings" in h) and "per share" not in h:
                results.append(span)
                break
    return results

# Limit to first 2 tables per document
def rule_limit_spans(doc):
    results = []
    financial_keywords = ["item 6", "item 7", "item 8", "selected financial",
                        "financial statement", "management's discussion"]
    for span in doc.get("texts", []):
        if span.get("label") != "table":
            continue
        path_text = span.get("structure", {}).get("path_text", "").lower()
        if not any(kw in path_text for kw in financial_keywords):
            continue
        cells = span.get("table_data", {}).get("cells", [])
        row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
        for h in row_headers:
            if ("net income" in h or "net earnings" in h or "net loss" in h) and "per share" not in h:
                results.append(span)
                break
        if len(results) >= 3:
            break
    return results

def test_rule(rule_fn, rule_name, docs, ground_truth):
    print(f"\n--- {rule_name} ---")
    hits = 0
    costs = []
    for doc_name in SAMPLED_DOCS:
        if doc_name not in docs:
            continue
        doc = docs[doc_name]
        answer = ground_truth.get(doc_name, "")
        spans = rule_fn(doc)
        retrieved_text = " ".join(s.get("text", "") for s in spans)
        hit = check_hit(retrieved_text, answer)
        cost = compute_cost(spans, doc)
        if hit:
            hits += 1
        costs.append(cost)
        status = "✓" if hit else "✗"
        print(f"  {status} {doc_name}: hit={hit}, cost={cost:.4f}, spans={len(spans)}")

    hit_rate = hits / len(SAMPLED_DOCS)
    avg_cost = sum(costs) / len(costs)
    print(f"  SUMMARY: hit_rate={hit_rate:.2%}, avg_cost={avg_cost:.4f}")
    return hit_rate, avg_cost

def main():
    docs = load_documents()
    ground_truth = load_ground_truth()

    print("=" * 80)
    print("TESTING TIGHTER RULES")
    print("=" * 80)

    rules = [
        (rule_original, "rule_original (Item 6, 7, 8)"),
        (rule_item8_only, "rule_item8_only"),
        (rule_item8_item6, "rule_item8_item6"),
        (rule_limit_spans, "rule_limit_spans (max 3)"),
    ]

    for rule_fn, rule_name in rules:
        test_rule(rule_fn, rule_name, docs, ground_truth)

if __name__ == "__main__":
    main()
