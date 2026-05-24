#!/usr/bin/env python3
"""Verify telephone rule with detailed analysis."""

import json
import re
from pathlib import Path

GROUND_TRUTH = {
    "AMCOR_2019_10K": "+44 117 9753200",
    "COSTCO_2017_10K": "(425) 313-8100",
    "BOEING_2018_10K": "(312) 544-2000",
    "AMAZON_2018_10K": "(206) 266-1000",
    "EBAY_2021_10K": "(408) 376-7108",
    "AMAZON_2016_10K": "(206) 266-1000",
    "CORNING_2022_10K": "607-974-9000",
    "NIKE_2021_10K": "(503) 671-6453",
    "LOCKHEEDMARTIN_2022_10K": "(301) 897-6000",
    "JOHNSON_JOHNSON_2022_10K": "(732) 524-0400",
}

DATA_DIR = Path("data/financebench/processing")

def count_tokens_simple(text):
    """Simple token count approximation."""
    return max(1, len(text.split()))

def load_doc(doc_name):
    path = DATA_DIR / f"{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

def rule_page1_phone_pattern(doc: dict) -> list[dict]:
    """Match page 1 spans containing phone number patterns."""
    try:
        results = []
        phone_patterns = [
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{3}-\d{3}-\d{4}',
            r'\+\d{2}\s+\d{3}\s+\d+',
        ]
        combined_pattern = '|'.join(phone_patterns)
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            if re.search(combined_pattern, text):
                results.append(span)
        return results
    except Exception:
        return []

def main():
    print("=" * 70)
    print("TELEPHONE NUMBER RULE VERIFICATION")
    print("=" * 70)

    results = []
    hits = 0
    total_cost = 0

    for doc_name, expected in GROUND_TRUTH.items():
        doc = load_doc(doc_name)
        spans = rule_page1_phone_pattern(doc)

        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))

        # Check if answer is in retrieved text
        expected_norm = re.sub(r'\s+', '', expected)
        retrieved_norm = re.sub(r'\s+', '', retrieved_text)
        hit = expected_norm in retrieved_norm or expected in retrieved_text

        if hit:
            hits += 1

        # Calculate cost
        retrieved_tokens = count_tokens_simple(retrieved_text)
        full_tokens = count_tokens_simple(full_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        status = "✓ HIT" if hit else "✗ MISS"
        print(f"\n{doc_name}: {status}")
        print(f"  Expected: {expected}")
        print(f"  Spans: {len(spans)}, Cost: {cost:.6f}")
        if not hit:
            print(f"  Retrieved: {retrieved_text[:200]}...")
        else:
            # Show the context around the phone number
            for s in spans:
                if expected in s.get("text", "") or expected_norm in re.sub(r'\s+', '', s.get("text", "")):
                    print(f"  Found in span: {s.get('text', '')[:100]}...")
                    break

        results.append({
            "doc_name": doc_name,
            "expected": expected,
            "hit": hit,
            "cost": cost,
            "spans_count": len(spans),
        })

    accuracy = hits / len(GROUND_TRUTH)
    avg_cost = total_cost / len(GROUND_TRUTH)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Hit Rate: {hits}/{len(GROUND_TRUTH)} = {accuracy:.2%}")
    print(f"Avg Cost Ratio: {avg_cost:.6f}")
    print(f"Target Cost: < 0.05 (5%)")
    print(f"Cost Met: {'YES' if avg_cost < 0.05 else 'NO'}")
    print(f"Accuracy Met (>= 95%): {'YES' if accuracy >= 0.95 else 'NO'}")

    return {
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "results": results
    }

if __name__ == "__main__":
    main()
