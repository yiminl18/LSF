#!/usr/bin/env python3
"""Refined test framework for long-term debt rules."""

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
        return True
    if answer == "0":
        return True

    answer_lower = str(answer).lower()
    text_lower = retrieved_text.lower()
    text_no_comma = text_lower.replace(",", "")

    # Check for exact substring
    if answer_lower in text_lower:
        return True

    # Check for numeric match
    numeric = extract_numeric_value(answer)
    if numeric and numeric in text_no_comma:
        return True

    # Handle thousands notation: $988,924,000 -> look for 988,924 or 988924
    # (when document shows "in thousands")
    if numeric:
        # Try as thousands (divide by 1000)
        try:
            val = float(numeric)
            thousands_val = int(val / 1000)
            if str(thousands_val) in text_no_comma:
                return True
        except:
            pass

    # Handle billions
    if "billion" in answer_lower:
        match = re.search(r'([\d.]+)\s*billion', answer_lower)
        if match:
            val = float(match.group(1))
            millions = int(val * 1000)
            if str(millions) in text_no_comma:
                return True
            # Also check formatted: 3.605 billion = 3,605 million
            formatted = f"{millions:,}"
            if formatted.replace(",", "") in text_no_comma:
                return True

    return False

# Define refined rules
def rule_debt_note_tables(doc):
    """Tables in debt-specific notes (Note 17, Note 14, etc.)."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("structure", {}).get("path_text", "").lower()
                    for kw in ["note 17", "note 14", "note 15", "note 16", ". debt"])]
    except Exception:
        return []

def rule_long_term_debt_tables(doc):
    """Tables containing 'long-term debt' or 'long term debt' keyword."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except Exception:
        return []

def rule_balance_sheet_item8(doc):
    """Balance sheet tables in Item 8 (first 3 pages of Item 8 section)."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            path_text = s.get("structure", {}).get("path_text", "").lower()
            text = s.get("text", "").lower()
            # Look for Item 8 balance sheets with debt info
            if ("item 8" in path_text or "financial statements" in path_text):
                if any(kw in text for kw in ["assets", "liabilities", "debt", "borrowing"]):
                    results.append(s)
        return results
    except Exception:
        return []

def rule_selected_financial_data(doc):
    """Tables in Item 6 Selected Financial Data with debt info."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("item 6" in s.get("structure", {}).get("path_text", "").lower() or
                 "selected financial" in s.get("structure", {}).get("path_text", "").lower()) and
                "debt" in s.get("text", "").lower()]
    except Exception:
        return []

def rule_10q_debt_tables(doc):
    """Debt tables for 10Q documents in Part I."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            path_text = s.get("structure", {}).get("path_text", "").lower()
            text = s.get("text", "").lower()
            # Part I Financial Information with debt
            if "part i" in path_text and ("debt" in text or "borrowing" in text):
                results.append(s)
        return results
    except Exception:
        return []

def rule_unsecured_notes_tables(doc):
    """Tables with unsecured notes / senior notes information."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("text", "").lower() for kw in [
                    "unsecured", "senior notes", "notes due", "term loan"
                ])]
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
                numeric = extract_numeric_value(answer)
                print(f"  Looking for: {answer} (numeric: {numeric})")

    # Summary
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_cost = sum(r["cost"] for r in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Total: {total}, Hits: {hits}, Accuracy: {hits/total:.2%}")
    print(f"Avg cost: {avg_cost:.4f}")

    return results

if __name__ == "__main__":
    print("Testing refined rules:")
    all_rules = [
        rule_debt_note_tables,
        rule_long_term_debt_tables,
        rule_balance_sheet_item8,
        rule_selected_financial_data,
        rule_10q_debt_tables,
        rule_unsecured_notes_tables,
    ]
    results = test_rules(all_rules)

    # Show per-rule stats
    print("\n" + "="*60)
    print("Per-rule statistics:")
    labels = load_labels()

    for rule in all_rules:
        hits = 0
        total_cost = 0
        count = 0
        for doc_name in DOCS:
            doc = load_doc(doc_name)
            if not doc:
                continue
            answer = labels.get(doc_name)
            spans = rule(doc)
            retrieved_text = "\n".join(s.get("text", "") for s in spans)
            full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
            ret_tokens = count_tokens(retrieved_text) if retrieved_text else 0
            full_tokens = count_tokens(full_text) if full_text else 1
            cost = ret_tokens / full_tokens if full_tokens > 0 else 0
            hit = check_hit(retrieved_text, answer)
            if hit:
                hits += 1
            total_cost += cost
            count += 1
        print(f"  {rule.__name__}: hits={hits}/{count}, avg_cost={total_cost/count:.4f}")
