#!/usr/bin/env python3
"""Test address rules for coverage and cost."""

import json
import os
import sys
import time
import importlib.util

START_TIME = time.time()

# Sampled documents
SAMPLED_DOCS = [
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
    "Pfizer_2023Q2_10Q",
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is the address of principal executive offices and ZIP code?"

# Load labels
with open("data/financebench/sample_mix_doc_labels.json") as f:
    labels = json.load(f)

# Build ground truth dict
ground_truth = {}
for doc_name in SAMPLED_DOCS:
    pdf_name = doc_name + ".pdf"
    if pdf_name in labels and QUESTION in labels[pdf_name]:
        ground_truth[doc_name] = labels[pdf_name][QUESTION]

# Load rules
RULES_DIR = "rules/agent/financebench_mix_doc_claude/what_is_the_address_of_principal_executive_offices_and_zip_c"
rules = []

for fname in os.listdir(RULES_DIR):
    if fname.endswith(".py") and fname.startswith("rule_"):
        fpath = os.path.join(RULES_DIR, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            if name.startswith("rule_"):
                rules.append((name, getattr(module, name)))

print(f"Loaded {len(rules)} rules: {[r[0] for r in rules]}")

# Test each document
hits = 0
total = 0
cost_sum = 0
cost_count = 0

results = {}

for doc_name in SAMPLED_DOCS:
    doc_path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        print(f"SKIP (missing): {doc_name}")
        continue

    with open(doc_path) as f:
        doc = json.load(f)

    answer = ground_truth.get(doc_name, "")
    if not answer:
        print(f"SKIP (no answer): {doc_name}")
        continue

    total += 1

    # Apply all rules and merge
    retrieved_spans = []
    for rule_name, rule_func in rules:
        spans = rule_func(doc)
        retrieved_spans.extend(spans)

    # Dedupe by text
    seen = set()
    unique_spans = []
    for s in retrieved_spans:
        text = s.get("text", "")
        if text not in seen:
            seen.add(text)
            unique_spans.append(s)

    # Concatenate retrieved text
    retrieved_text = " ".join(s.get("text", "") for s in unique_spans)

    # Full doc text
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))

    # Cost ratio
    if len(full_text) > 0:
        cost = len(retrieved_text) / len(full_text)
        cost_sum += cost
        cost_count += 1
    else:
        cost = 0

    # Substring match for hit
    answer_lower = answer.lower()
    retrieved_lower = retrieved_text.lower()

    # For address, we need flexible matching - check key components
    # Extract address parts
    import re
    zip_match = re.search(r'\d{5}(?:-\d{4})?', answer)
    zip_code = zip_match.group() if zip_match else ""

    # Split address to get street part
    address_parts = answer.split(",")
    street_part = address_parts[0].strip().lower() if address_parts else ""

    has_street = street_part and street_part in retrieved_lower
    has_zip = zip_code and zip_code in retrieved_lower

    # For UK addresses like AMCOR, check for BS30 or similar
    uk_postcode = re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', answer, re.I)
    if uk_postcode:
        uk_match = uk_postcode.group().replace(" ", "")
        has_zip = has_zip or uk_match.lower() in retrieved_lower.replace(" ", "")

    is_hit = has_street and (has_zip or not zip_code)

    if is_hit:
        hits += 1
        status = "HIT"
    else:
        status = "MISS"

    results[doc_name] = {
        "status": status,
        "answer": answer,
        "has_street": has_street,
        "has_zip": has_zip,
        "cost": cost,
        "retrieved_len": len(retrieved_text),
        "full_len": len(full_text),
    }

    print(f"{status}: {doc_name} (cost={cost:.4f}, has_street={has_street}, has_zip={has_zip})")
    if status == "MISS":
        print(f"  ANSWER: {answer}")
        print(f"  RETRIEVED: {retrieved_text[:300]}...")

print(f"\n{'='*60}")
print(f"Total: {total}")
print(f"Hits: {hits}")
print(f"Hit Rate: {hits/total:.2%}" if total > 0 else "N/A")
print(f"Avg Cost: {cost_sum/cost_count:.4f}" if cost_count > 0 else "N/A")
print(f"Time: {time.time() - START_TIME:.2f}s")
