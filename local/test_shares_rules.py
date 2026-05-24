#!/usr/bin/env python3
"""Test script for shares outstanding rules."""

import json
import re
import time
import tiktoken

# Question and documents
QUESTION = "How many shares of common stock were outstanding as of the cover-page reference date?"
DOC_NAMES = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
]

# Load ground truth
with open("data/financebench/sample_doc_labels.json") as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in DOC_NAMES:
    doc_key = f"{doc_name}.pdf"
    if doc_key in labels:
        ground_truth[doc_name] = labels[doc_key][QUESTION]

print("Ground truth:")
for doc, ans in ground_truth.items():
    print(f"  {doc}: {ans}")

# Load documents
docs = {}
for doc_name in DOC_NAMES:
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    with open(path) as f:
        docs[doc_name] = json.load(f)

# Rule 1: Page 1-2 spans with "shares" and "outstanding" keywords
def rule_cover_shares_outstanding(doc: dict) -> list[dict]:
    """Match page 1-2 spans mentioning shares outstanding."""
    try:
        import re
        results = []
        for span in doc.get("texts", []):
            page = span.get("page_no", 0)
            if page not in (1, 2):
                continue
            text = span.get("text", "").lower()
            # Match spans that mention shares outstanding
            if ("shares" in text and "outstanding" in text) or \
               ("common stock" in text and "outstanding" in text):
                results.append(span)
        return results
    except Exception:
        return []

# Rule 2: Page 1-2 spans that contain large numbers following a shares label
def rule_cover_shares_number(doc: dict) -> list[dict]:
    """Match page 1-2 spans with numbers that follow shares outstanding labels."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            page = span.get("page_no", 0)
            if page not in (1, 2):
                continue
            text = span.get("text", "").strip()
            # Check if this span contains a large number (could be multiple)
            numbers = re.findall(r'[\d,]{7,}', text)  # At least 7 chars for millions
            if not numbers:
                continue
            # Check if any number is large enough (> 1 million)
            has_large_num = False
            for num_str in numbers:
                try:
                    num = int(num_str.replace(",", ""))
                    if num >= 1_000_000:
                        has_large_num = True
                        break
                except:
                    continue
            if not has_large_num:
                continue
            # Check if any previous span on same page mentions "shares" and "outstanding"
            for j in range(max(0, i-10), i):
                prev = texts[j]
                if prev.get("page_no") != page:
                    continue
                prev_text = prev.get("text", "").lower()
                if "shares" in prev_text and ("outstanding" in prev_text or "common stock" in prev_text):
                    results.append(span)
                    break
        return results
    except Exception:
        return []

# Compute cost
enc = tiktoken.get_encoding("cl100k_base")

def get_doc_tokens(doc):
    all_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
    return len(enc.encode(all_text))

def get_spans_tokens(spans):
    text = " ".join(s.get("text", "") for s in spans)
    return len(enc.encode(text))

# Test each rule
print("\n" + "="*60)
print("Testing Rule 1: rule_cover_shares_outstanding")
print("="*60)

for doc_name in DOC_NAMES:
    doc = docs[doc_name]
    gt = ground_truth[doc_name]

    spans = rule_cover_shares_outstanding(doc)
    combined_text = " ".join(s.get("text", "") for s in spans)

    # Check if answer is found
    gt_clean = gt.replace(",", "")
    hit = gt_clean in combined_text.replace(",", "")

    doc_tokens = get_doc_tokens(doc)
    span_tokens = get_spans_tokens(spans)
    cost = span_tokens / doc_tokens if doc_tokens > 0 else 0

    status = "HIT" if hit else "MISS"
    print(f"{doc_name}: {status}, spans={len(spans)}, cost={cost:.4f}")
    if not hit and spans:
        print(f"  Retrieved text preview: {combined_text[:200]}...")
        print(f"  Looking for: {gt}")

print("\n" + "="*60)
print("Testing Rule 2: rule_cover_shares_number")
print("="*60)

for doc_name in DOC_NAMES:
    doc = docs[doc_name]
    gt = ground_truth[doc_name]

    spans = rule_cover_shares_number(doc)
    combined_text = " ".join(s.get("text", "") for s in spans)

    gt_clean = gt.replace(",", "")
    hit = gt_clean in combined_text.replace(",", "")

    doc_tokens = get_doc_tokens(doc)
    span_tokens = get_spans_tokens(spans)
    cost = span_tokens / doc_tokens if doc_tokens > 0 else 0

    status = "HIT" if hit else "MISS"
    print(f"{doc_name}: {status}, spans={len(spans)}, cost={cost:.4f}")

# Combined rules
print("\n" + "="*60)
print("Testing Combined Rules (union)")
print("="*60)

total_hits = 0
total_cost = 0

for doc_name in DOC_NAMES:
    doc = docs[doc_name]
    gt = ground_truth[doc_name]

    spans1 = rule_cover_shares_outstanding(doc)
    spans2 = rule_cover_shares_number(doc)

    # Deduplicate by text
    seen = set()
    combined_spans = []
    for s in spans1 + spans2:
        key = s.get("text", "")
        if key not in seen:
            seen.add(key)
            combined_spans.append(s)

    combined_text = " ".join(s.get("text", "") for s in combined_spans)

    gt_clean = gt.replace(",", "")
    hit = gt_clean in combined_text.replace(",", "")
    if hit:
        total_hits += 1

    doc_tokens = get_doc_tokens(doc)
    span_tokens = get_spans_tokens(combined_spans)
    cost = span_tokens / doc_tokens if doc_tokens > 0 else 0
    total_cost += cost

    status = "HIT" if hit else "MISS"
    print(f"{doc_name}: {status}, spans={len(combined_spans)}, cost={cost:.4f}")
    if not hit:
        print(f"  Looking for: {gt}")
        print(f"  Retrieved: {combined_text[:300]}...")

print(f"\nSummary: {total_hits}/{len(DOC_NAMES)} hits ({100*total_hits/len(DOC_NAMES):.1f}%)")
print(f"Average cost: {total_cost/len(DOC_NAMES):.4f}")
