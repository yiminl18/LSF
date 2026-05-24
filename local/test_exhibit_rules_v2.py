#!/usr/bin/env python3
"""Test exhibit rules for material agreement question - v2."""

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
    patterns = [
        f"data/financebench/processing/{doc_name}_reconstructed.json",
        f"data/financebench/processing/{doc_name.replace('-', '_')}_reconstructed.json",
    ]
    for path in patterns:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def rule_exhibit_index_broad(doc: dict) -> list:
    """Match spans in exhibit-related sections (Item 15, Item 6, Item 9.01) or containing exhibit patterns."""
    try:
        import re
        results = []

        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")

            # Check if in exhibit section (path-based)
            in_exhibit_section = any(x in path for x in ["item 15", "item 6", "item 9.01", "exhibit"])

            # Check if text contains exhibit section header
            has_exhibit_section_in_text = "item 9.01" in text_lower or "financial statements and exhibits" in text_lower

            # Check for exhibit number patterns in text
            # Matches: "Exhibit 10.1", "10.1 Credit Agreement", "99.1. Press release", "104 Cover Page"
            exhibit_pattern = re.search(r'\b(exhibit\s*)?\d+(\.\d+)?\s*(:|\.|\s)', text_lower)

            if in_exhibit_section or has_exhibit_section_in_text:
                results.append(span)
            elif exhibit_pattern and any(x in path for x in ["item 8", "item 5", "financial"]):
                # For 8-K docs, exhibits may be under other item paths
                results.append(span)

        return results
    except Exception:
        return []


def rule_exhibit_section_tables_and_text(doc: dict) -> list:
    """Match tables and relevant text in exhibit sections."""
    try:
        import re
        results = []

        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")

            # For 10-K: tables in Item 15 section
            if "item 15" in path and label == "table":
                results.append(span)
                continue

            # For 10-Q: tables or text in Item 6 section with exhibit references
            if "item 6" in path and ("exhibit" in text_lower or label == "table"):
                results.append(span)
                continue

            # For 8-K: Item 9.01 section
            if "item 9.01" in path or "9.01" in path:
                # Include section header and exhibit descriptions
                if "exhibit" in text_lower or label == "section_header":
                    results.append(span)
                # Include text spans that look like exhibit entries (e.g., "99.1. Press release")
                elif re.match(r'^\d+\.\d*\.?\s+\w', text):
                    results.append(span)
                elif label == "table":
                    results.append(span)
                continue

            # Catch tables containing exhibit info
            if label == "table" and "9.01" in text_lower and "exhibit" in text_lower:
                results.append(span)
                continue

        return results
    except Exception:
        return []


def rule_exhibit_tables_with_fallback(doc: dict) -> list:
    """Primary: exhibit tables. Fallback: text spans in Item 9.01 for 8-K."""
    try:
        import re
        results = []
        found_exhibit_table = False

        # First pass: find tables in exhibit sections
        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")

            if label == "table":
                # Tables in Item 15 (10-K), Item 6 (10-Q), or containing exhibit references
                if any(x in path for x in ["item 15", "item 6"]) and "exhibit" in text_lower:
                    results.append(span)
                    found_exhibit_table = True
                elif "exhibit" in path:
                    results.append(span)
                    found_exhibit_table = True
                # Tables containing Item 9.01 info (8-K)
                elif "9.01" in text_lower and "exhibit" in text_lower:
                    results.append(span)
                    found_exhibit_table = True

        # Second pass for 8-K: if no exhibit tables found, look for text in Item 9.01
        if not found_exhibit_table:
            for span in doc.get("texts", []):
                text = span.get("text", "")
                text_lower = text.lower()
                path = span.get("structure", {}).get("path_text", "").lower()

                if "9.01" in path:
                    # Include text that looks like exhibit entries
                    if re.match(r'^\d+\.\d*\.?\s+\w', text) or "exhibit" in text_lower:
                        results.append(span)
                    elif "financial statements and exhibits" in text_lower:
                        results.append(span)

        return results
    except Exception:
        return []


def rule_final_combined(doc: dict) -> list:
    """Combined rule: exhibit tables for 10-K/10-Q, Item 9.01 section for 8-K."""
    try:
        import re
        results = []
        is_8k = "_8K" in doc.get("doc_name", "")

        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")

            if is_8k:
                # For 8-K: capture Item 9.01 section
                if "9.01" in path or "financial statements and exhibits" in path:
                    if "exhibit" in text_lower or re.match(r'^\d+\.\d*\.?\s*\w', text) or label == "table":
                        results.append(span)
                # Also capture tables containing Item 9.01
                elif label == "table" and "9.01" in text_lower:
                    results.append(span)
            else:
                # For 10-K/10-Q: capture exhibit tables
                if label == "table":
                    if any(x in path for x in ["item 15", "item 6", "exhibit"]):
                        if "exhibit" in text_lower:
                            results.append(span)
                    elif "exhibit" in path and "exhibit" in text_lower:
                        results.append(span)

        return results
    except Exception:
        return []


def test_rules():
    rules = [
        rule_exhibit_index_broad,
        rule_exhibit_section_tables_and_text,
        rule_exhibit_tables_with_fallback,
        rule_final_combined,
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
        details = []

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
                # For "none listed" answers, we just need exhibit section to confirm no material agreements
                hit = len(spans) > 0
            else:
                # Check if key parts of the answer appear in retrieved text
                hit = answer.lower() in retrieved_text.lower()
                if not hit:
                    # Try partial match - check exhibit number
                    match = re.search(r'exhibit\s*(\d+\.?\d*)', answer, re.IGNORECASE)
                    if match:
                        exhibit_num = match.group(1)
                        # Check for "Exhibit 10.1" or just "10.1"
                        hit = exhibit_num in retrieved_text
                    # Also try description match
                    if not hit and ":" in answer:
                        desc_part = answer.split(":")[1].strip()[:30]
                        hit = desc_part.lower() in retrieved_text.lower()

            if hit:
                hits += 1
                status = "HIT"
            else:
                status = "MISS"

            details.append((status, doc_name, cost, len(spans), answer))

        for status, doc_name, cost, num_spans, answer in details:
            print(f"  [{status}] {doc_name}: cost={cost:.4f}, spans={num_spans}")
            if status == "MISS" and answer:
                print(f"        Expected: {answer[:80]}")

        accuracy = hits / valid_docs if valid_docs > 0 else 0
        avg_cost = total_cost / valid_docs if valid_docs > 0 else 0

        print(f"\n  Summary: accuracy={accuracy:.2%}, avg_cost={avg_cost:.4f}, valid_docs={valid_docs}")

if __name__ == "__main__":
    test_rules()
