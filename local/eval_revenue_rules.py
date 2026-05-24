#!/usr/bin/env python3
"""Evaluate revenue retrieval rules for total revenue question."""

import json
import re
import tiktoken

# Configuration
DOC_NAMES = [
    "AMCOR_2019_10K", "COSTCO_2017_10K", "BOEING_2018_10K", "AMAZON_2018_10K",
    "EBAY_2021_10K", "AMAZON_2016_10K", "CORNING_2022_10K", "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K", "JOHNSON_JOHNSON_2022_10K"
]

GROUND_TRUTH = {
    "AMCOR_2019_10K": "9,458.2 million",
    "COSTCO_2017_10K": "126,172",
    "BOEING_2018_10K": "$101,127 million",
    "AMAZON_2018_10K": "$232,887 million",
    "EBAY_2021_10K": "$10,420 million",
    "AMAZON_2016_10K": "$135,987 million",
    "CORNING_2022_10K": "14,189 million",
    "NIKE_2021_10K": "$44,538 million",
    "LOCKHEEDMARTIN_2022_10K": "$65,984 million",
    "JOHNSON_JOHNSON_2022_10K": "$94.9 billion"
}

# GT number patterns for substring matching
GT_NUMS = {
    "AMCOR_2019_10K": ["9,458", "9458"],
    "COSTCO_2017_10K": ["126,172", "126172"],
    "BOEING_2018_10K": ["101,127", "101127"],
    "AMAZON_2018_10K": ["232,887", "232887"],
    "EBAY_2021_10K": ["10,420", "10420"],
    "AMAZON_2016_10K": ["135,987", "135987"],
    "CORNING_2022_10K": ["14,189", "14189"],
    "NIKE_2021_10K": ["44,538", "44538"],
    "LOCKHEEDMARTIN_2022_10K": ["65,984", "65984"],
    "JOHNSON_JOHNSON_2022_10K": ["94,943", "94943"]  # 94.9 billion = 94943 million
}

def load_doc(doc_name):
    """Load a document JSON."""
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

def get_full_text(doc):
    """Get full document text for cost calculation."""
    return " ".join(span.get("text", "") for span in doc.get("texts", []))

def get_retrieved_text(spans):
    """Get concatenated text from retrieved spans."""
    return " ".join(span.get("text", "") for span in spans)

def count_tokens(text):
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def check_hit(text, doc_name):
    """Check if GT value appears in text (substring match)."""
    patterns = GT_NUMS.get(doc_name, [])
    text_clean = text.replace(",", "").replace(" ", "")
    for p in patterns:
        p_clean = p.replace(",", "").replace(" ", "")
        if p_clean in text_clean:
            return True
    return False

def evaluate_rule(rule_func, docs):
    """Evaluate a single rule across all documents."""
    results = []
    for doc_name, doc in docs.items():
        spans = rule_func(doc)
        retrieved_text = get_retrieved_text(spans)
        full_text = get_full_text(doc)

        hit = check_hit(retrieved_text, doc_name)
        retrieved_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0

        results.append({
            "doc_name": doc_name,
            "hit": hit,
            "cost": cost,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens
        })

    hit_rate = sum(r["hit"] for r in results) / len(results)
    avg_cost = sum(r["cost"] for r in results) / len(results)

    return {
        "hit_rate": hit_rate,
        "avg_cost": avg_cost,
        "results": results
    }

def evaluate_merged_rules(rule_funcs, docs):
    """Evaluate merged rules (union of spans) across all documents."""
    results = []
    for doc_name, doc in docs.items():
        all_spans = []
        seen_ids = set()
        for rule_func in rule_funcs:
            spans = rule_func(doc)
            for span in spans:
                span_id = id(span)
                if span_id not in seen_ids:
                    seen_ids.add(span_id)
                    all_spans.append(span)

        retrieved_text = get_retrieved_text(all_spans)
        full_text = get_full_text(doc)

        hit = check_hit(retrieved_text, doc_name)
        retrieved_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0

        results.append({
            "doc_name": doc_name,
            "hit": hit,
            "cost": cost,
            "num_spans": len(all_spans),
            "retrieved_tokens": retrieved_tokens
        })

    hit_rate = sum(r["hit"] for r in results) / len(results)
    avg_cost = sum(r["cost"] for r in results) / len(results)

    return {
        "hit_rate": hit_rate,
        "avg_cost": avg_cost,
        "results": results
    }

# Test rules

def rule_revenue_table_financial_sections(doc):
    """Match tables with revenue/sales row headers in financial sections."""
    try:
        results = []
        revenue_keywords = ["revenue", "net sales", "total sales", "sales to customer"]
        path_keywords = ["item 6", "item 7", "item 8", "selected financial",
                        "management's discussion", "financial statement",
                        "consolidated statement", "statement of income",
                        "statement of earnings", "statement of operations"]

        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue

            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]

            for h in row_headers:
                if any(kw in h for kw in revenue_keywords):
                    results.append(span)
                    break

            if len(results) >= 3:
                break

        return results
    except Exception:
        return []

def rule_consolidated_income_statement(doc):
    """Match consolidated income/earnings statement tables."""
    try:
        results = []
        path_keywords = ["consolidated statement", "statement of income",
                        "statement of earnings", "statement of operations"]

        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue

            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue

            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]

            # Check if it looks like an income statement (has revenue and cost/expense)
            has_revenue = any(any(kw in h for kw in ["revenue", "sales"]) for h in row_headers)
            has_cost = any(any(kw in h for kw in ["cost", "expense", "gross"]) for h in row_headers)

            if has_revenue and has_cost:
                results.append(span)

            if len(results) >= 2:
                break

        return results
    except Exception:
        return []

if __name__ == "__main__":
    # Load all documents
    docs = {}
    for doc_name in DOC_NAMES:
        docs[doc_name] = load_doc(doc_name)

    print("=== Rule Evaluation ===\n")

    # Test rule 1
    print("Rule 1: revenue_table_financial_sections")
    result = evaluate_rule(rule_revenue_table_financial_sections, docs)
    print(f"  Hit rate: {result['hit_rate']:.2%}")
    print(f"  Avg cost: {result['avg_cost']:.4f}")
    missed = [r["doc_name"] for r in result["results"] if not r["hit"]]
    if missed:
        print(f"  Missed: {missed}")
    print()

    # Test rule 2
    print("Rule 2: consolidated_income_statement")
    result = evaluate_rule(rule_consolidated_income_statement, docs)
    print(f"  Hit rate: {result['hit_rate']:.2%}")
    print(f"  Avg cost: {result['avg_cost']:.4f}")
    missed = [r["doc_name"] for r in result["results"] if not r["hit"]]
    if missed:
        print(f"  Missed: {missed}")
    print()

    # Test merged
    print("Merged (both rules):")
    result = evaluate_merged_rules([rule_revenue_table_financial_sections, rule_consolidated_income_statement], docs)
    print(f"  Hit rate: {result['hit_rate']:.2%}")
    print(f"  Avg cost: {result['avg_cost']:.4f}")
    missed = [r["doc_name"] for r in result["results"] if not r["hit"]]
    if missed:
        print(f"  Missed: {missed}")

    # Show per-doc details
    print("\nPer-document details:")
    for r in result["results"]:
        status = "HIT" if r["hit"] else "MISS"
        print(f"  {r['doc_name']}: {status}, cost={r['cost']:.4f}, spans={r['num_spans']}")
