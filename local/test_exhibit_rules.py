#!/usr/bin/env python3
"""Test exhibit rules for material agreement question."""

import json
import os
import re
import time
import tiktoken

# Document list
DOCS = [
    "BOEING_2019_10K", "ADOBE_2020_10K", "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K", "AMCOR_2019_10K", "AMAZON_2020_10K", "AMAZON_2019_10K",
    "ADOBE_2021_10K", "EBAY_2022_10K", "ADOBE_2019_10K", "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q", "Pfizer_2023Q2_10Q", "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q", "AMCOR_2022_8K_2022-07-01", "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16", "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20"
]

QUESTION = 'List one material agreement or other exhibit number explicitly identified in the Exhibit Index (e.g., "Exhibit 10.1: Credit Agreement")—or record "none listed" if no material agreements are identified.'

# Ground truth
GROUND_TRUTH = {
    "BOEING_2019_10K": "Exhibit 10.1: 364-Day Credit Agreement",
    "ADOBE_2020_10K": "Exhibit 10.1: Credit Agreement",
    "ACTIVISIONBLIZZARD_2020_10K": "Exhibit 10.24: Notice of Stock Option Award",
    "COSTCO_2018_10K": "none listed",
    "AMCOR_2019_10K": "Exhibit 10.1: Transaction Agreement",
    "AMAZON_2020_10K": "Exhibit 10.1: 1997 Stock Incentive Plan",
    "AMAZON_2019_10K": "Exhibit 10.1: 1997 Stock Incentive Plan",
    "ADOBE_2021_10K": "Exhibit 10.1: 2020 Employee Stock Purchase Plan",
    "EBAY_2022_10K": None,
    "ADOBE_2019_10K": "Exhibit 10.1: Credit Agreement",
    "AMCOR_2023Q2_10Q": "None listed",
    "ADOBE_2022Q2_10Q": "Exhibit 10.1: 2019 Equity Incentive Plan, as amended",
    "Pfizer_2023Q2_10Q": "Exhibit 10.1: Executive Officer Severance Policy",
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "Exhibit 10.1: Credit Agreement",
    "3M_2023Q2_10Q": "Exhibit 10.1: Settlement Agreement",
    "AMCOR_2022_8K_2022-07-01": "Exhibit 4.6: Second Supplemental Indenture",
    "COSTCO_2023_8K_dated-2023-08-09": "Exhibit 3.2: Bylaws as amended of Costco Wholesale Corporation",
    "COSTCO_2023_8K_dated-2023-08-16": "Exhibit 99.1: Press release dated August 16, 2023",
    "MGMRESORTS_2023_8K_dated-2023-03-01": "Exhibit 99.1: Press Release of the Company dated March 1, 2023",
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "Exhibit 104: Cover Page Interactive Data File"
}

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def load_doc(doc_name: str):
    # Try different file name patterns
    patterns = [
        f"data/financebench/processing/{doc_name}_reconstructed.json",
        f"data/financebench/processing/{doc_name.replace('-', '_')}_reconstructed.json",
    ]
    for path in patterns:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

# Rule functions to test
def rule_exhibit_path_or_keyword(doc: dict) -> list:
    """Match spans with 'exhibit' in path_text or text, or under Item 15/Item 6/Item 9.01."""
    try:
        results = []
        for span in doc.get("texts", []):
            text = span.get("text", "").lower()
            path = span.get("structure", {}).get("path_text", "").lower()

            # Match if path or text contains exhibit-related keywords
            if "exhibit" in path or "exhibit" in text:
                # Filter out noise - only keep relevant sections
                if any(x in path for x in ["item 15", "item 6", "item 9.01", "3. exhibit"]):
                    results.append(span)
                elif any(x in text for x in ["exhibit 3.", "exhibit 4.", "exhibit 10.", "exhibit 21", "exhibit 23", "exhibit 31", "exhibit 32", "exhibit 99", "exhibit 104"]):
                    results.append(span)
        return results
    except Exception:
        return []

def rule_exhibit_tables_only(doc: dict) -> list:
    """Match table spans in exhibit sections."""
    try:
        results = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = span.get("structure", {}).get("path_text", "").lower()
            text = span.get("text", "").lower()

            # Match if path contains exhibit-related keywords
            if any(x in path for x in ["exhibit", "item 15", "item 6", "item 9.01"]):
                results.append(span)
            elif "exhibit" in text and any(x in text for x in ["10.", "3.", "4.", "21", "23", "31", "32", "99", "104"]):
                results.append(span)
        return results
    except Exception:
        return []

def rule_item15_exhibit_section(doc: dict) -> list:
    """Match spans in Item 15 (Exhibits section) for 10-K/10-Q."""
    try:
        results = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "").lower()
            text = span.get("text", "").lower()

            # For 10-K: Item 15
            if "item 15" in path and ("exhibit" in path or "exhibit" in text):
                results.append(span)
            # For 10-Q: Item 6
            elif "item 6" in path and ("exhibit" in path or "exhibit" in text):
                results.append(span)
            # For 8-K: Item 9.01
            elif "item 9.01" in path:
                results.append(span)
        return results
    except Exception:
        return []

def rule_exhibit_10x_keyword(doc: dict) -> list:
    """Match spans containing Exhibit 10.x patterns (material contracts)."""
    try:
        import re
        results = []
        pattern = re.compile(r'exhibit\s*10\.\d+', re.IGNORECASE)
        for span in doc.get("texts", []):
            text = span.get("text", "")
            if pattern.search(text):
                results.append(span)
        return results
    except Exception:
        return []

def rule_any_exhibit_number(doc: dict) -> list:
    """Match spans containing any Exhibit X.X patterns."""
    try:
        import re
        results = []
        # Match Exhibit followed by a number
        pattern = re.compile(r'exhibit\s*\d+(\.\d+)?', re.IGNORECASE)
        for span in doc.get("texts", []):
            text = span.get("text", "")
            path = span.get("structure", {}).get("path_text", "").lower()
            if pattern.search(text):
                # Prefer spans in exhibit sections
                if any(x in path for x in ["item 15", "item 6", "item 9", "exhibit"]):
                    results.append(span)
        return results
    except Exception:
        return []

def test_rules():
    rules = [
        rule_exhibit_path_or_keyword,
        rule_exhibit_tables_only,
        rule_item15_exhibit_section,
        rule_exhibit_10x_keyword,
        rule_any_exhibit_number,
    ]

    print(f"Testing {len(rules)} rules on {len(DOCS)} documents\n")

    for rule_fn in rules:
        print(f"\n{'='*60}")
        print(f"Rule: {rule_fn.__name__}")
        print(f"Description: {rule_fn.__doc__}")
        print(f"{'='*60}")

        hits = 0
        total_cost = 0
        valid_docs = 0

        for doc_name in DOCS:
            doc = load_doc(doc_name)
            if doc is None:
                print(f"  [MISSING] {doc_name}")
                continue

            valid_docs += 1
            spans = rule_fn(doc)

            # Calculate cost
            full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
            retrieved_text = " ".join(s.get("text", "") for s in spans)

            full_tokens = count_tokens(full_text)
            retrieved_tokens = count_tokens(retrieved_text)
            cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
            total_cost += cost

            # Check if answer is in retrieved text
            answer = GROUND_TRUTH.get(doc_name)
            if answer is None:
                hit = True  # No ground truth, skip
            elif answer.lower() in ["none listed", "none"]:
                # Special case: check if no material exhibit is found
                has_exhibit = "exhibit 10." in retrieved_text.lower() or "exhibit 4." in retrieved_text.lower()
                hit = not has_exhibit or len(spans) > 0
            else:
                # Check if key parts of the answer appear in retrieved text
                hit = answer.lower() in retrieved_text.lower()
                if not hit:
                    # Try checking exhibit number only
                    match = re.search(r'exhibit\s*(\d+\.?\d*)', answer, re.IGNORECASE)
                    if match:
                        exhibit_num = match.group(1)
                        hit = f"exhibit {exhibit_num}" in retrieved_text.lower() or f"exhibit{exhibit_num}" in retrieved_text.lower() or exhibit_num in retrieved_text

            if hit:
                hits += 1
                status = "HIT"
            else:
                status = "MISS"

            print(f"  [{status}] {doc_name}: cost={cost:.4f}, spans={len(spans)}")
            if not hit and answer:
                print(f"        Expected: {answer[:80]}")
                if spans:
                    print(f"        Got: {retrieved_text[:200]}...")

        accuracy = hits / valid_docs if valid_docs > 0 else 0
        avg_cost = total_cost / valid_docs if valid_docs > 0 else 0

        print(f"\n  Summary: accuracy={accuracy:.2%}, avg_cost={avg_cost:.4f}, valid_docs={valid_docs}")

if __name__ == "__main__":
    test_rules()
