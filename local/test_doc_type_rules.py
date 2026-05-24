#!/usr/bin/env python3
"""Test document type rules for SEC form identification."""
import json
import os
import re

# Sampled documents (excluding missing files)
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
    # "Pfizer_2023Q2_10Q",  # Missing
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    # "MGMRESORTS_2023_8K_dated-2023-03-01",  # Missing
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What document type or SEC form is this (e.g. 10-K, 10-Q, 8-K, earnings release, annual report)?"
DATA_DIR = "data/financebench/processing"
LABELS_FILE = "data/financebench/sample_mix_doc_labels.json"

def load_doc(doc_name):
    path = os.path.join(DATA_DIR, f"{doc_name}_reconstructed.json")
    with open(path, "r") as f:
        return json.load(f)

def load_labels():
    with open(LABELS_FILE, "r") as f:
        return json.load(f)

def rule_page1_form_type(doc):
    """Match page 1 spans containing FORM 10-K, 10-Q, or 8-K."""
    try:
        results = []
        form_patterns = [r"form\s*10-k", r"form\s*10-q", r"form\s*8-k"]
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower()
            text_span = s.get("text_span", "").lower()
            combined = text + " " + text_span
            if any(re.search(p, combined) for p in form_patterns):
                results.append(s)
        return results
    except Exception:
        return []

def rule_page1_form_type_tight(doc):
    """Match page 1 spans where text field (not text_span) contains FORM 10-K/10-Q/8-K."""
    try:
        results = []
        form_patterns = [r"form\s*10-k", r"form\s*10-q", r"form\s*8-k"]
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower()
            if any(re.search(p, text) for p in form_patterns):
                results.append(s)
        return results
    except Exception:
        return []

def count_tokens(text):
    """Rough token count (words / 0.75)."""
    return max(1, int(len(text.split()) / 0.75))

def get_full_doc_text(doc):
    return " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in doc.get("texts", []))

def get_span_text(spans):
    return " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in spans)

def test_rule(rule_func, rule_name, ground_truth):
    """Test a rule and return results."""
    results = []
    for doc_name in SAMPLED_DOCS:
        doc = load_doc(doc_name)
        spans = rule_func(doc)

        gt = ground_truth.get(doc_name, "")
        retrieved_text = get_span_text(spans)
        full_text = get_full_doc_text(doc)

        # Substring match check
        gt_lower = gt.lower()
        retrieved_lower = retrieved_text.lower()

        # Check if form type is in retrieved text
        hit = False
        if "10-k" in gt_lower and "10-k" in retrieved_lower:
            hit = True
        elif "10-q" in gt_lower and "10-q" in retrieved_lower:
            hit = True
        elif "8-k" in gt_lower and "8-k" in retrieved_lower:
            hit = True

        # Cost calculation
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0

        results.append({
            "doc": doc_name,
            "gt": gt,
            "num_spans": len(spans),
            "hit": hit,
            "cost": cost,
            "retrieved_preview": retrieved_text[:200] if retrieved_text else "[EMPTY]"
        })
    return results

def main():
    labels = load_labels()

    # Build ground truth dict
    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        pdf_name = f"{doc_name}.pdf"
        if pdf_name in labels:
            gt = labels[pdf_name].get(QUESTION, "")
            ground_truth[doc_name] = gt

    print(f"Testing {len(SAMPLED_DOCS)} documents")
    print(f"Ground truth loaded for {len(ground_truth)} docs")

    # Test both rules
    for rule_func, rule_name in [(rule_page1_form_type, "rule_page1_form_type"),
                                   (rule_page1_form_type_tight, "rule_page1_form_type_tight")]:
        print()
        print("="*80)
        print(f"TESTING: {rule_name}")
        print("="*80)

        results = test_rule(rule_func, rule_name, ground_truth)

        hits = sum(1 for r in results if r["hit"])
        avg_cost = sum(r["cost"] for r in results) / len(results)

        print(f"\nHit Rate: {hits}/{len(results)} = {hits/len(results)*100:.1f}%")
        print(f"Avg Cost: {avg_cost*100:.3f}%")
        print()

        # Show details for each doc
        for r in results:
            status = "✓" if r["hit"] else "✗"
            print(f"{status} {r['doc']}: gt='{r['gt']}', spans={r['num_spans']}, cost={r['cost']*100:.3f}%")
            if not r["hit"]:
                print(f"  Retrieved: {r['retrieved_preview']}")

    return

    results = []
    for doc_name in SAMPLED_DOCS:
        doc = load_doc(doc_name)
        spans = rule_page1_form_type(doc)

        gt = ground_truth.get(doc_name, "")
        retrieved_text = get_span_text(spans)
        full_text = get_full_doc_text(doc)

        # Substring match check
        gt_lower = gt.lower()
        retrieved_lower = retrieved_text.lower()

        # Check if form type is in retrieved text
        hit = False
        if "10-k" in gt_lower and "10-k" in retrieved_lower:
            hit = True
        elif "10-q" in gt_lower and "10-q" in retrieved_lower:
            hit = True
        elif "8-k" in gt_lower and "8-k" in retrieved_lower:
            hit = True

        # Cost calculation
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0

        results.append({
            "doc": doc_name,
            "gt": gt,
            "num_spans": len(spans),
            "hit": hit,
            "cost": cost,
            "retrieved_preview": retrieved_text[:200] if retrieved_text else "[EMPTY]"
        })

    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)

    hits = sum(1 for r in results if r["hit"])
    avg_cost = sum(r["cost"] for r in results) / len(results)

    print(f"\nHit Rate: {hits}/{len(results)} = {hits/len(results)*100:.1f}%")
    print(f"Avg Cost: {avg_cost*100:.3f}%")
    print()

    # Show details for each doc
    for r in results:
        status = "✓" if r["hit"] else "✗"
        print(f"{status} {r['doc']}: gt='{r['gt']}', spans={r['num_spans']}, cost={r['cost']*100:.3f}%")
        if not r["hit"]:
            print(f"  Retrieved: {r['retrieved_preview']}")

    # Show missed docs
    missed = [r for r in results if not r["hit"]]
    if missed:
        print("\n" + "="*80)
        print("MISSED DOCUMENTS - Need analysis")
        print("="*80)
        for r in missed:
            print(f"\n{r['doc']}:")
            print(f"  Ground truth: {r['gt']}")
            print(f"  Retrieved: {r['retrieved_preview']}")

if __name__ == "__main__":
    main()
