#!/usr/bin/env python3
"""Test framework for long-term debt rules."""

import json
import re
from pathlib import Path

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
except ImportError:
    def count_tokens(text): return len(text.split())

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
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
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
    text = str(answer).replace(",", "").replace("$", "")
    match = re.search(r'([\d.]+)', text)
    if match:
        return match.group(1)
    return None

def check_hit(retrieved_text, answer):
    """Check if answer is found in retrieved text (substring match)."""
    if answer is None:
        return True  # null answer is always a "hit" (we just need relevant context)
    if answer == "0":
        return True  # 0 means no debt info needed, accepting any retrieval

    answer_lower = str(answer).lower()
    text_lower = retrieved_text.lower()

    # Check for exact substring
    if answer_lower in text_lower:
        return True

    # Check for numeric match
    numeric = extract_numeric_value(answer)
    if numeric and numeric in text_lower.replace(",", ""):
        return True

    # Special patterns
    # "$3.605 billion" -> "3,605" or "3.605"
    if "billion" in answer_lower:
        match = re.search(r'([\d.]+)\s*billion', answer_lower)
        if match:
            val = float(match.group(1))
            # Try finding as millions (3.605 billion = 3605 million)
            millions = int(val * 1000)
            if str(millions) in text_lower.replace(",", ""):
                return True
            # Try as decimal
            if f"{val:.3f}".replace(".", "") in text_lower.replace(",", "").replace(".", ""):
                return True

    return False

# Define rules to test
def rule_item8_financial_tables(doc):
    """Tables in Item 8 Financial Statements section."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("structure", {}).get("path_text", "").lower()
                    for kw in ["item 8", "item 6", "financial statements", "financial information"])]
    except Exception:
        return []

def rule_debt_keyword_tables(doc):
    """Tables containing 'debt' in text or path_text."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("debt" in s.get("text", "").lower() or
                 "debt" in s.get("structure", {}).get("path_text", "").lower())]
    except Exception:
        return []

def rule_balance_sheet_tables(doc):
    """Balance sheet tables with debt or liabilities info."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            text = s.get("text", "").lower()
            path_text = s.get("structure", {}).get("path_text", "").lower()
            # Look for balance sheet markers
            if any(kw in text or kw in path_text for kw in [
                "balance sheet", "financial position",
                "long-term debt", "total debt", "long term debt"
            ]):
                results.append(s)
        return results
    except Exception:
        return []

def rule_part1_financial_tables(doc):
    """Part I financial tables for 10Q documents."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("structure", {}).get("path_text", "").lower()
                    for kw in ["part i", "condensed consolidated"])]
    except Exception:
        return []

def rule_8k_cover_page(doc):
    """Cover page for 8K documents to signal document type."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("page_no") == 1 and
                "8-k" in s.get("text", "").lower()]
    except Exception:
        return []

# Test function
def test_rules(rules, verbose=True):
    labels = load_labels()
    results = []

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if not doc:
            if verbose:
                print(f"SKIP: {doc_name} not found")
            continue

        answer = labels.get(doc_name)

        # Apply all rules and union results
        all_spans = []
        seen = set()
        for rule in rules:
            spans = rule(doc)
            for s in spans:
                span_id = id(s)
                if span_id not in seen:
                    seen.add(span_id)
                    all_spans.append(s)

        # Compute retrieved text
        retrieved_text = "\n".join(s.get("text", "") for s in all_spans)
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))

        # Compute cost
        ret_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text) if full_text else 1
        cost = ret_tokens / full_tokens if full_tokens > 0 else 0

        # Check hit
        hit = check_hit(retrieved_text, answer)

        results.append({
            "doc": doc_name,
            "answer": answer,
            "hit": hit,
            "cost": cost,
            "spans": len(all_spans),
            "ret_tokens": ret_tokens,
            "full_tokens": full_tokens
        })

        if verbose:
            status = "HIT" if hit else "MISS"
            print(f"{status}: {doc_name}, answer={answer}, spans={len(all_spans)}, cost={cost:.4f}")
            if not hit and answer and answer != "0":
                # Show what's missing
                numeric = extract_numeric_value(answer)
                print(f"  Looking for: {answer} (numeric: {numeric})")
                # Check first few tables
                for i, s in enumerate(all_spans[:3]):
                    print(f"  Span {i}: {s.get('text', '')[:100]}...")

    # Summary
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_cost = sum(r["cost"] for r in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Total: {total}, Hits: {hits}, Accuracy: {hits/total:.2%}")
    print(f"Avg cost: {avg_cost:.4f}")

    return results

if __name__ == "__main__":
    # Test individual rules
    print("Testing rule_item8_financial_tables:")
    test_rules([rule_item8_financial_tables])

    print("\n" + "="*60)
    print("Testing rule_debt_keyword_tables:")
    test_rules([rule_debt_keyword_tables])

    print("\n" + "="*60)
    print("Testing combined rules:")
    test_rules([
        rule_item8_financial_tables,
        rule_debt_keyword_tables,
        rule_balance_sheet_tables,
        rule_part1_financial_tables
    ])
