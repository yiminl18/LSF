#!/usr/bin/env python3
"""Optimized rules for long-term debt with lower cost."""

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
    if answer is None or answer == "0":
        return None
    text = str(answer).replace(",", "").replace("$", "")
    match = re.search(r'([\d.]+)', text)
    if match:
        return match.group(1)
    return None

def check_hit(retrieved_text, answer):
    if answer is None:
        return True
    if answer == "0":
        return True

    answer_lower = str(answer).lower()
    text_lower = retrieved_text.lower()
    text_no_comma = text_lower.replace(",", "")

    if answer_lower in text_lower:
        return True

    numeric = extract_numeric_value(answer)
    if numeric and numeric in text_no_comma:
        return True

    if numeric:
        try:
            val = float(numeric)
            thousands_val = int(val / 1000)
            if str(thousands_val) in text_no_comma:
                return True
        except:
            pass

    if "billion" in answer_lower:
        match = re.search(r'([\d.]+)\s*billion', answer_lower)
        if match:
            val = float(match.group(1))
            millions = int(val * 1000)
            if str(millions) in text_no_comma:
                return True

    return False

# Optimized rules - minimal set for coverage
def rule_long_term_debt_tables(doc):
    """Tables containing 'long-term debt' keyword - the main rule."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except Exception:
        return []

def rule_debt_note_tables_tight(doc):
    """Tables in debt notes - tighter filter."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("structure", {}).get("path_text", "").lower()
                    for kw in [". debt"]) and
                "note" in s.get("structure", {}).get("path_text", "").lower()]
    except Exception:
        return []

def rule_senior_notes_tables(doc):
    """Tables with senior notes/term loan details."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                any(kw in s.get("text", "").lower() for kw in [
                    "senior notes", "term loan", "notes due"
                ]) and
                "debt" in s.get("text", "").lower()]
    except Exception:
        return []

def rule_balance_sheet_debt_rows(doc):
    """Balance sheet tables specifically with debt line items."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            text = s.get("text", "").lower()
            path_text = s.get("structure", {}).get("path_text", "").lower()

            # Only get balance sheets with debt info
            is_balance_sheet = any(kw in text or kw in path_text for kw in [
                "balance sheet", "financial position", "assets", "liabilities"
            ])
            has_debt_info = any(kw in text for kw in [
                "long-term debt", "long term debt", "total debt",
                "current portion of long-term"
            ])

            if is_balance_sheet and has_debt_info:
                results.append(s)
        return results
    except Exception:
        return []

def test_rules(rules, verbose=True):
    labels = load_labels()
    results = []

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if not doc:
            continue

        answer = labels.get(doc_name)

        all_spans = []
        seen = set()
        for rule in rules:
            spans = rule(doc)
            for s in spans:
                span_id = id(s)
                if span_id not in seen:
                    seen.add(span_id)
                    all_spans.append(s)

        retrieved_text = "\n".join(s.get("text", "") for s in all_spans)
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))

        ret_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text) if full_text else 1
        cost = ret_tokens / full_tokens if full_tokens > 0 else 0

        hit = check_hit(retrieved_text, answer)

        results.append({
            "doc": doc_name,
            "answer": answer,
            "hit": hit,
            "cost": cost,
            "spans": len(all_spans),
        })

        if verbose:
            status = "HIT" if hit else "MISS"
            print(f"{status}: {doc_name}, answer={answer}, spans={len(all_spans)}, cost={cost:.4f}")

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_cost = sum(r["cost"] for r in results) / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Total: {total}, Hits: {hits}, Accuracy: {hits/total:.2%}")
    print(f"Avg cost: {avg_cost:.4f}")

    return results

if __name__ == "__main__":
    # Test with most selective rules first
    print("Testing minimal rule set:")
    rules = [
        rule_long_term_debt_tables,
        rule_debt_note_tables_tight,
        rule_senior_notes_tables,
        rule_balance_sheet_debt_rows,
    ]
    test_rules(rules)

    print("\n" + "="*60)
    print("Testing just long_term_debt_tables + debt_note_tables_tight:")
    test_rules([rule_long_term_debt_tables, rule_debt_note_tables_tight])

    print("\n" + "="*60)
    print("Testing just long_term_debt_tables:")
    test_rules([rule_long_term_debt_tables])
