#!/usr/bin/env python3
"""Test script for stock exchange rules development."""

import json
import tiktoken
import re

# Ground truth for stock exchange question
GROUND_TRUTH = {
    "BOEING_2019_10K": "New York Stock Exchange",
    "ADOBE_2020_10K": "NASDAQ",
    "ACTIVISIONBLIZZARD_2020_10K": "The Nasdaq Global Select Market",
    "COSTCO_2018_10K": "The NASDAQ Global Select Market",
    "AMCOR_2019_10K": "The New York Stock Exchange",
    "AMAZON_2020_10K": "Nasdaq Global Select Market",
    "AMAZON_2019_10K": "Nasdaq Global Select Market",
    "ADOBE_2021_10K": "NASDAQ",
    "EBAY_2022_10K": "The Nasdaq Global Select Market",
    "ADOBE_2019_10K": "NASDAQ",
    "AMCOR_2023Q2_10Q": "New York Stock Exchange",
    "ADOBE_2022Q2_10Q": "NASDAQ",
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "The Nasdaq Global Select Market",
    "3M_2023Q2_10Q": "New York Stock Exchange",
    "AMCOR_2022_8K_2022-07-01": "New York Stock Exchange",
    "COSTCO_2023_8K_dated-2023-08-09": "NASDAQ",
    "COSTCO_2023_8K_dated-2023-08-16": "NASDAQ",
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "New York Stock Exchange"
}

def load_doc(doc_name):
    """Load document JSON."""
    with open(f"data/financebench/processing/{doc_name}_reconstructed.json") as f:
        return json.load(f)

def get_full_text_tokens(doc):
    """Get token count for full document text."""
    enc = tiktoken.get_encoding("cl100k_base")
    full_text = " ".join(s.get("text", "") for s in doc["texts"])
    return len(enc.encode(full_text))

def get_span_tokens(spans):
    """Get token count for spans."""
    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(s.get("text", "") for s in spans)
    return len(enc.encode(text))

def contains_answer(spans, answer):
    """Check if any span contains the answer (case-insensitive)."""
    text = " ".join(s.get("text", "") for s in spans).lower()
    return answer.lower() in text

# Rule 1: Page 1 spans containing stock exchange keywords
def rule_page1_exchange_keywords(doc):
    """Match page 1 spans with stock exchange keywords."""
    try:
        keywords = ["nasdaq", "new york stock exchange", "nyse", "stock exchange"]
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            any(kw in s.get("text", "").lower() for kw in keywords)
        ]
    except Exception:
        return []

# Rule 2: Page 1 spans with path containing exchange or trading symbol
def rule_page1_exchange_path(doc):
    """Match page 1 spans with exchange-related path."""
    try:
        keywords = ["exchange", "trading symbol"]
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            any(kw in s.get("structure", {}).get("path_text", "").lower() for kw in keywords)
        ]
    except Exception:
        return []

# Rule 3: Tables on page 1 containing exchange info
def rule_page1_exchange_table(doc):
    """Match tables on page 1 with exchange column headers."""
    try:
        result = []
        for s in doc.get("texts", []):
            if s.get("page_no") == 1 and s.get("label") == "table":
                text = s.get("text", "").lower()
                if "exchange" in text or "nasdaq" in text or "new york" in text:
                    result.append(s)
        return result
    except Exception:
        return []

# Combined rule: All page 1 spans with exchange signals
def rule_page1_combined_exchange(doc):
    """Match page 1 spans with any exchange-related signals."""
    try:
        keywords = ["nasdaq", "new york stock exchange", "stock exchange", "nyse"]
        path_keywords = ["exchange", "trading symbol"]
        result = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text_lower = s.get("text", "").lower()
            path_lower = s.get("structure", {}).get("path_text", "").lower()

            # Check text keywords
            if any(kw in text_lower for kw in keywords):
                result.append(s)
                continue
            # Check path keywords
            if any(kw in path_lower for kw in path_keywords):
                result.append(s)
                continue
            # Check table content
            if s.get("label") == "table" and ("exchange" in text_lower or "nasdaq" in text_lower):
                result.append(s)
        return result
    except Exception:
        return []

def evaluate_rules(rules):
    """Evaluate rules on all documents."""
    results = {}

    for doc_name, answer in GROUND_TRUTH.items():
        doc = load_doc(doc_name)
        full_tokens = get_full_text_tokens(doc)

        # Apply all rules and merge
        all_spans = []
        for rule in rules:
            spans = rule(doc)
            all_spans.extend(spans)

        # Deduplicate by span index
        seen = set()
        unique_spans = []
        for s in all_spans:
            idx = doc["texts"].index(s)
            if idx not in seen:
                seen.add(idx)
                unique_spans.append(s)

        span_tokens = get_span_tokens(unique_spans)
        cost = span_tokens / full_tokens if full_tokens > 0 else 0
        hit = contains_answer(unique_spans, answer)

        results[doc_name] = {
            "hit": hit,
            "cost": cost,
            "num_spans": len(unique_spans),
            "span_tokens": span_tokens,
            "full_tokens": full_tokens
        }

    # Compute summary
    hit_rate = sum(1 for r in results.values() if r["hit"]) / len(results)
    avg_cost = sum(r["cost"] for r in results.values()) / len(results)

    return results, hit_rate, avg_cost

def main():
    # Test individual rules
    rules_to_test = [
        ("page1_exchange_keywords", [rule_page1_exchange_keywords]),
        ("page1_exchange_path", [rule_page1_exchange_path]),
        ("page1_exchange_table", [rule_page1_exchange_table]),
        ("page1_combined_exchange", [rule_page1_combined_exchange]),
    ]

    for name, rules in rules_to_test:
        print(f"\n=== Testing: {name} ===")
        results, hit_rate, avg_cost = evaluate_rules(rules)
        print(f"Hit rate: {hit_rate:.2%} ({sum(1 for r in results.values() if r['hit'])}/{len(results)})")
        print(f"Avg cost: {avg_cost:.4f}")

        # Show misses
        misses = [doc for doc, r in results.items() if not r["hit"]]
        if misses:
            print(f"Misses: {misses}")

if __name__ == "__main__":
    main()
