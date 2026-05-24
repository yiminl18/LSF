#!/usr/bin/env python3
"""Compute per-rule statistics."""

import json
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

def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def rule_long_term_debt_tables(doc):
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except:
        return []

def rule_total_obligations_tables(doc):
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                "long-term obligations" in s.get("text", "").lower()]
    except:
        return []

rules = [
    ("rule_long_term_debt_tables", rule_long_term_debt_tables),
    ("rule_total_obligations_tables", rule_total_obligations_tables),
]

for rule_name, rule_fn in rules:
    total_cost = 0
    coverage = 0

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if not doc:
            continue

        spans = rule_fn(doc)
        if spans:
            coverage += 1

        retrieved_text = "\n".join(s.get("text", "") for s in spans)
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))

        ret_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text) if full_text else 1
        cost = ret_tokens / full_tokens

        total_cost += cost

    avg_cost = total_cost / len(DOCS)
    print(f"{rule_name}:")
    print(f"  Coverage: {coverage}/{len(DOCS)} docs")
    print(f"  Avg cost: {avg_cost:.4f}")
