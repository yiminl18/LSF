#!/usr/bin/env python3
"""Test and evaluate net income retrieval rules with improved hit detection."""

import json
import re
import time
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text):
        return len(text) // 4

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
    """Extract numeric value from answer, handling billion/million."""
    text = text.lower().replace(",", "").replace(" ", "")
    text = text.replace("$", "")

    # Extract number and unit
    match = re.search(r'([\d.]+)\s*(billion|million|b|m)?', text)
    if match:
        num = float(match.group(1))
        unit = match.group(2) or ""
        if "billion" in unit or unit == "b":
            return num * 1000  # Convert to millions
        elif "million" in unit or unit == "m":
            return num
        else:
            return num
    return None

def check_hit(retrieved_text, answer):
    """Check if answer appears in retrieved text, handling billion/million conversion."""
    retrieved_clean = retrieved_text.lower().replace(",", "").replace(" ", "")
    answer_clean = answer.lower().replace(",", "").replace(" ", "")

    # Direct substring match
    if answer_clean in retrieved_clean:
        return True

    # Extract primary number from answer
    answer_nums = re.findall(r'[\d,]+\.?\d*', answer)
    for num in answer_nums:
        num_clean = num.replace(",", "")
        if num_clean in retrieved_text.replace(",", ""):
            return True

    # Handle billion/million conversion
    # If answer is in billions (e.g., "17.9 billion"), look for the millions equivalent
    if "billion" in answer.lower():
        answer_value = extract_numeric_value(answer)
        if answer_value:
            # Look for the value in millions (with some tolerance)
            # e.g., 17.9 billion = 17,900 million, but actual might be 17,941
            target_low = int(answer_value * 0.99)  # 1% tolerance
            target_high = int(answer_value * 1.01)

            # Find any number in the range in the text
            text_nums = re.findall(r'[\d,]+', retrieved_text)
            for tn in text_nums:
                try:
                    tn_val = int(tn.replace(",", ""))
                    if target_low <= tn_val <= target_high:
                        return True
                except ValueError:
                    continue

    return False

def compute_cost(retrieved_spans, doc):
    full_text = " ".join(span.get("text", "") for span in doc.get("texts", []))
    retrieved_text = " ".join(span.get("text", "") for span in retrieved_spans)
    full_tokens = count_tokens(full_text)
    retrieved_tokens = count_tokens(retrieved_text)
    if full_tokens == 0:
        return 0.0
    return retrieved_tokens / full_tokens

# ============== RULE DEFINITIONS ==============

def rule_table_net_income_header(doc: dict) -> list[dict]:
    """Match tables with row headers containing 'net income' or 'net earnings'."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if ("net income" in h or "net earnings" in h or "net loss" in h) and "per share" not in h:
                    results.append(span)
                    break
        return results
    except Exception:
        return []

def rule_table_net_income_item8(doc: dict) -> list[dict]:
    """Match tables in Item 8 Financial Statements with net income row headers."""
    try:
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
    except Exception:
        return []

def rule_table_income_statement(doc: dict) -> list[dict]:
    """Match tables in income statement sections with net income row headers."""
    try:
        results = []
        income_keywords = ["income statement", "statement of income", "statement of operation",
                          "consolidated statement of earnings", "statements of earnings",
                          "statement of comprehensive income"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in income_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if ("net income" in h or "net earnings" in h) and "per share" not in h:
                    results.append(span)
                    break
        return results
    except Exception:
        return []

def rule_table_selected_financial_data(doc: dict) -> list[dict]:
    """Match tables in Selected Financial Data section with net income row headers."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if "item 6" not in path_text and "selected financial" not in path_text:
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if ("net income" in h or "net earnings" in h) and "per share" not in h:
                    results.append(span)
                    break
        return results
    except Exception:
        return []

# Primary combined rule for final use
def rule_table_net_income_financial_sections(doc: dict) -> list[dict]:
    """Match tables with net income/earnings headers in financial sections (Item 6, 7, 8)."""
    try:
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
    except Exception:
        return []

ALL_RULES = [
    ("rule_table_net_income_header", rule_table_net_income_header),
    ("rule_table_net_income_item8", rule_table_net_income_item8),
    ("rule_table_income_statement", rule_table_income_statement),
    ("rule_table_selected_financial_data", rule_table_selected_financial_data),
    ("rule_table_net_income_financial_sections", rule_table_net_income_financial_sections),
]

def test_rules():
    docs = load_documents()
    ground_truth = load_ground_truth()

    print("=" * 80)
    print("TESTING NET INCOME RULES (with billion/million handling)")
    print("=" * 80)

    for rule_name, rule_fn in ALL_RULES:
        print(f"\n--- {rule_name} ---")
        hits = 0
        costs = []
        coverage = []

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
                coverage.append(doc_name)
            costs.append(cost)

            print(f"  {doc_name}: hit={hit}, cost={cost:.4f}, spans={len(spans)}, answer={answer[:30]}...")

        hit_rate = hits / len(SAMPLED_DOCS)
        avg_cost = sum(costs) / len(costs) if costs else 0
        print(f"  SUMMARY: hit_rate={hit_rate:.2%}, avg_cost={avg_cost:.4f}, coverage={len(coverage)}")

    # Test merged rules (union of all)
    print("\n" + "=" * 80)
    print("MERGED RULES (UNION)")
    print("=" * 80)

    hits = 0
    costs = []
    for doc_name in SAMPLED_DOCS:
        if doc_name not in docs:
            continue
        doc = docs[doc_name]
        answer = ground_truth.get(doc_name, "")

        all_spans = []
        seen_indices = set()
        for _, rule_fn in ALL_RULES:
            for span in rule_fn(doc):
                idx = id(span)
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    all_spans.append(span)

        retrieved_text = " ".join(s.get("text", "") for s in all_spans)
        hit = check_hit(retrieved_text, answer)
        cost = compute_cost(all_spans, doc)

        if hit:
            hits += 1
        costs.append(cost)

        print(f"  {doc_name}: hit={hit}, cost={cost:.4f}, spans={len(all_spans)}")

    hit_rate = hits / len(SAMPLED_DOCS)
    avg_cost = sum(costs) / len(costs) if costs else 0
    print(f"\n  MERGED SUMMARY: hit_rate={hit_rate:.2%}, avg_cost={avg_cost:.4f}")

if __name__ == "__main__":
    test_rules()
