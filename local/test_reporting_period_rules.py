#!/usr/bin/env python3
"""Test reporting period rules and measure accuracy/cost."""

import json
import os
import re
import tiktoken

# Map from expected doc names to actual filenames
DOC_MAP = {
    "BOEING_2019_10K": "BOEING_2019_10K",
    "ADOBE_2020_10K": "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K": "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K": "COSTCO_2018_10K",
    "AMCOR_2019_10K": "AMCOR_2019_10K",
    "AMAZON_2020_10K": "AMAZON_2020_10K",
    "AMAZON_2019_10K": "AMAZON_2019_10K",
    "ADOBE_2021_10K": "ADOBE_2021_10K",
    "EBAY_2022_10K": "EBAY_2022_10K",
    "ADOBE_2019_10K": "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q": "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q": "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q": None,  # Not available
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q": "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01": "AMCOR_2022_8K_dated-2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09": "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16": "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01": None,  # Not available
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "FOOTLOCKER_2022_8K_dated-2022-05-20",
}

QUESTION = "What is the reporting period covered by this document (e.g. fiscal year ended, quarter ended, or event date)?"

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def load_doc(doc_name):
    actual_name = DOC_MAP.get(doc_name, doc_name)
    if actual_name is None:
        return None
    path = f"data/financebench/processing/{actual_name}_reconstructed.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    gt = {}
    for doc_name in DOC_MAP.keys():
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            gt[doc_name] = labels[key][QUESTION]
    return gt

# ==================== RULES ====================

def rule_page1_period_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans containing reporting period keywords."""
    try:
        keywords = [
            "fiscal year ended",
            "quarterly period ended",
            "date of report",
            "date of earliest event",
        ]
        results = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower()
            text_span = s.get("text_span", "").lower()
            combined = text + " " + text_span
            if any(kw in combined for kw in keywords):
                results.append(s)
        return results
    except Exception:
        return []

def rule_page12_form_header_period(doc: dict) -> list[dict]:
    """Match page 1-2 spans under FORM 10-K/10-Q/8-K path with period keywords."""
    try:
        form_paths = ["form 10-k", "form 10-q", "form 8-k"]
        period_keywords = [
            "fiscal year ended",
            "quarterly period ended",
            "date of report",
            "date of earliest event",
        ]
        results = []
        for s in doc.get("texts", []):
            page = s.get("page_no", 0)
            if page > 2:
                continue
            path = s.get("structure", {}).get("path_text", "").lower()
            if not any(fp in path for fp in form_paths):
                continue
            text = s.get("text", "").lower()
            text_span = s.get("text_span", "").lower()
            combined = text + " " + text_span
            if any(kw in combined for kw in period_keywords):
                results.append(s)
        return results
    except Exception:
        return []

def rule_page1_for_the_period(doc: dict) -> list[dict]:
    """Match page 1 spans starting with 'For the fiscal' or 'For the quarterly'."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower().strip()
            if text.startswith("for the fiscal") or text.startswith("for the quarterly"):
                results.append(s)
        return results
    except Exception:
        return []

def rule_page1_date_of_report(doc: dict) -> list[dict]:
    """Match page 1 spans containing 'date of report' for 8-K documents."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            text = s.get("text", "").lower()
            text_span = s.get("text_span", "").lower()
            combined = text + " " + text_span
            if "date of report" in combined or "date of earliest event" in combined:
                results.append(s)
        return results
    except Exception:
        return []

# ==================== EVALUATION ====================

def get_full_doc_tokens(doc):
    """Get total tokens in document."""
    all_text = " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in doc.get("texts", []))
    return len(enc.encode(all_text))

def get_retrieved_tokens(spans):
    """Get tokens in retrieved spans."""
    all_text = " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in spans)
    return len(enc.encode(all_text))

def check_answer_in_spans(spans, answer):
    """Check if answer is in retrieved spans (substring match)."""
    answer_lower = answer.lower()
    for s in spans:
        combined = s.get("text", "").lower() + " " + s.get("text_span", "").lower()
        if answer_lower in combined:
            return True
        # Also check date extraction
        date_match = re.search(r'(\w+\s+\d+,?\s+\d{4})', answer)
        if date_match and date_match.group(1).lower() in combined:
            return True
    return False

def evaluate_rule(rule_func, docs, ground_truth):
    """Evaluate a single rule."""
    hits = 0
    total = 0
    costs = []

    for doc_name, answer in ground_truth.items():
        doc = docs.get(doc_name)
        if doc is None:
            continue

        total += 1
        spans = rule_func(doc)

        # Check hit
        if check_answer_in_spans(spans, answer):
            hits += 1

        # Calculate cost
        full_tokens = get_full_doc_tokens(doc)
        retrieved_tokens = get_retrieved_tokens(spans)
        if full_tokens > 0:
            costs.append(retrieved_tokens / full_tokens)

    hit_rate = hits / total if total > 0 else 0
    avg_cost = sum(costs) / len(costs) if costs else 0

    return {
        "hits": hits,
        "total": total,
        "hit_rate": hit_rate,
        "avg_cost": avg_cost,
        "costs": costs,
    }

def evaluate_merged_rules(rules, docs, ground_truth):
    """Evaluate union of all rules."""
    hits = 0
    total = 0
    costs = []
    details = []

    for doc_name, answer in ground_truth.items():
        doc = docs.get(doc_name)
        if doc is None:
            continue

        total += 1

        # Apply all rules and take union
        all_spans = []
        seen_ids = set()
        for rule_func in rules:
            for span in rule_func(doc):
                span_id = id(span)
                if span_id not in seen_ids:
                    seen_ids.add(span_id)
                    all_spans.append(span)

        # Check hit
        hit = check_answer_in_spans(all_spans, answer)
        if hit:
            hits += 1

        # Calculate cost
        full_tokens = get_full_doc_tokens(doc)
        retrieved_tokens = get_retrieved_tokens(all_spans)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        costs.append(cost)

        details.append({
            "doc": doc_name,
            "hit": hit,
            "answer": answer,
            "num_spans": len(all_spans),
            "cost": cost,
        })

    hit_rate = hits / total if total > 0 else 0
    avg_cost = sum(costs) / len(costs) if costs else 0

    return {
        "hits": hits,
        "total": total,
        "hit_rate": hit_rate,
        "avg_cost": avg_cost,
        "details": details,
    }

def main():
    # Load data
    ground_truth = load_labels()
    docs = {}
    for doc_name in DOC_MAP.keys():
        doc = load_doc(doc_name)
        if doc:
            docs[doc_name] = doc

    print(f"Loaded {len(docs)} documents, {len(ground_truth)} ground truth answers")
    print()

    # Define rules to test
    rules_to_test = [
        ("rule_page1_period_keywords", rule_page1_period_keywords),
        ("rule_page12_form_header_period", rule_page12_form_header_period),
        ("rule_page1_for_the_period", rule_page1_for_the_period),
        ("rule_page1_date_of_report", rule_page1_date_of_report),
    ]

    # Test each rule individually
    print("=" * 70)
    print("INDIVIDUAL RULE RESULTS")
    print("=" * 70)
    for name, rule_func in rules_to_test:
        result = evaluate_rule(rule_func, docs, ground_truth)
        print(f"\n{name}:")
        print(f"  Hit rate: {result['hits']}/{result['total']} = {result['hit_rate']:.2%}")
        print(f"  Avg cost: {result['avg_cost']:.4f} ({result['avg_cost']*100:.2f}%)")

    # Test merged rules
    print()
    print("=" * 70)
    print("MERGED RULES RESULT")
    print("=" * 70)

    # Find best single rule first
    best_rule = rules_to_test[0]
    best_result = evaluate_rule(best_rule[1], docs, ground_truth)

    for name, rule_func in rules_to_test[1:]:
        result = evaluate_rule(rule_func, docs, ground_truth)
        if result["hit_rate"] > best_result["hit_rate"]:
            best_rule = (name, rule_func)
            best_result = result
        elif result["hit_rate"] == best_result["hit_rate"] and result["avg_cost"] < best_result["avg_cost"]:
            best_rule = (name, rule_func)
            best_result = result

    print(f"\nBest single rule: {best_rule[0]}")
    print(f"  Hit rate: {best_result['hit_rate']:.2%}, Avg cost: {best_result['avg_cost']:.4f}")

    # Check if we need to combine rules
    if best_result["hit_rate"] < 0.95:
        print("\nTrying rule combinations...")
        all_rules = [r[1] for r in rules_to_test]
        merged = evaluate_merged_rules(all_rules, docs, ground_truth)
        print(f"\nAll rules merged:")
        print(f"  Hit rate: {merged['hits']}/{merged['total']} = {merged['hit_rate']:.2%}")
        print(f"  Avg cost: {merged['avg_cost']:.4f} ({merged['avg_cost']*100:.2f}%)")

        # Show misses
        if merged["hit_rate"] < 1.0:
            print("\nMisses:")
            for d in merged["details"]:
                if not d["hit"]:
                    print(f"  {d['doc']}: {d['answer']}")
    else:
        print("\nSingle rule achieves >= 95% hit rate, testing with details...")
        merged = evaluate_merged_rules([best_rule[1]], docs, ground_truth)
        print(f"\nDetails:")
        for d in merged["details"]:
            status = "HIT" if d["hit"] else "MISS"
            print(f"  [{status}] {d['doc']}: cost={d['cost']:.4f}, spans={d['num_spans']}")

if __name__ == "__main__":
    main()
