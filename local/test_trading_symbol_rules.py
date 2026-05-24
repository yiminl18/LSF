#!/usr/bin/env python3
"""Test and evaluate trading symbol rules."""

import json
import os
import tiktoken
from typing import Callable

# Sampled documents
SAMPLED_DOCS = [
    'BOEING_2019_10K',
    'ADOBE_2020_10K',
    'ACTIVISIONBLIZZARD_2020_10K',
    'COSTCO_2018_10K',
    'AMCOR_2019_10K',
    'AMAZON_2020_10K',
    'AMAZON_2019_10K',
    'ADOBE_2021_10K',
    'EBAY_2022_10K',
    'ADOBE_2019_10K',
    'AMCOR_2023Q2_10Q',
    'ADOBE_2022Q2_10Q',
    'Pfizer_2023Q2_10Q',
    'ACTIVSIONBLIZZARD_2023Q2_10Q',
    '3M_2023Q2_10Q',
    'AMCOR_2022_8K_2022-07-01',
    'COSTCO_2023_8K_dated-2023-08-09',
    'COSTCO_2023_8K_dated-2023-08-16',
    'MGMRESORTS_2023_8K_dated-2023-03-01',
    'FOOTLOCKER_2022_8K_dated-2022-05-20'
]

QUESTION = "What is/are the trading symbol(s) and listing exchange(s)?"

def load_ground_truth():
    """Load ground truth answers."""
    with open('data/financebench/sample_mix_doc_labels.json', 'r') as f:
        labels = json.load(f)

    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        pdf_name = doc_name + ".pdf"
        if pdf_name in labels and QUESTION in labels[pdf_name]:
            ans = labels[pdf_name][QUESTION]
            # Flatten complex answers
            if isinstance(ans, list):
                # Multiple entries - extract all symbols and exchanges
                parts = []
                for item in ans:
                    if isinstance(item, dict):
                        sym = item.get('symbol') or item.get('Trading Symbol')
                        exch = item.get('exchange') or item.get('Exchange')
                        if sym and exch:
                            parts.append(f"{sym}, {exch}")
                ground_truth[doc_name] = "; ".join(parts) if parts else str(ans)
            else:
                ground_truth[doc_name] = str(ans)
    return ground_truth

def load_doc(doc_name):
    """Load a document JSON."""
    path = f'data/financebench/processing/{doc_name}_reconstructed.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def count_tokens(text):
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_full_doc_text(doc):
    """Get full document text."""
    return " ".join(s.get("text", "") for s in doc.get("texts", []))

def get_retrieved_text(spans):
    """Get concatenated text from spans."""
    return " ".join(s.get("text", "") for s in spans)

def check_substring_hit(retrieved_text, answer):
    """Check if answer appears in retrieved text (case-insensitive)."""
    retrieved_lower = retrieved_text.lower()

    # Handle complex answers
    if ";" in answer:
        # Multiple symbols - check each
        parts = answer.split(";")
        for part in parts:
            # Extract symbol (usually first part before comma)
            symbol = part.strip().split(",")[0].strip()
            if symbol.lower() not in retrieved_lower:
                return False
        return True
    else:
        # Single answer - check key parts
        parts = answer.lower().replace(",", " ").replace("—", " ").replace("-", " ").split()
        # Check if both symbol and exchange appear
        key_parts = [p for p in parts if len(p) > 2]  # Filter very short parts
        hits = sum(1 for p in key_parts if p in retrieved_lower)
        return hits >= len(key_parts) * 0.5  # At least 50% of key parts match

def evaluate_rule(rule_func: Callable, verbose=False):
    """Evaluate a single rule."""
    ground_truth = load_ground_truth()

    results = {
        'hits': 0,
        'misses': 0,
        'total_cost': 0,
        'doc_results': []
    }

    for doc_name in SAMPLED_DOCS:
        doc = load_doc(doc_name)
        if doc is None:
            if verbose:
                print(f"  {doc_name}: SKIPPED (file not found)")
            continue

        if doc_name not in ground_truth:
            if verbose:
                print(f"  {doc_name}: SKIPPED (no ground truth)")
            continue

        answer = ground_truth[doc_name]
        spans = rule_func(doc)
        retrieved = get_retrieved_text(spans)

        full_text = get_full_doc_text(doc)
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved)

        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        hit = check_substring_hit(retrieved, answer)

        results['total_cost'] += cost
        if hit:
            results['hits'] += 1
        else:
            results['misses'] += 1

        results['doc_results'].append({
            'doc': doc_name,
            'hit': hit,
            'cost': cost,
            'spans_returned': len(spans),
            'answer': answer,
            'retrieved_preview': retrieved[:300] if not hit else ''
        })

        if verbose:
            status = "✓ HIT" if hit else "✗ MISS"
            print(f"  {doc_name}: {status}, cost={cost:.4f}, spans={len(spans)}")
            if not hit:
                print(f"    Answer: {answer}")
                print(f"    Retrieved: {retrieved[:200]}...")

    total = results['hits'] + results['misses']
    results['hit_rate'] = results['hits'] / total if total > 0 else 0
    results['avg_cost'] = results['total_cost'] / total if total > 0 else 0

    return results


# Test rules
def rule_page1_trading_symbol_table(doc: dict) -> list[dict]:
    """Match page 1 tables containing trading symbol column header."""
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            s.get("label") == "table" and
            "trading symbol" in s.get("text", "").lower()
        ]
    except Exception:
        return []

def rule_page1_securities_section(doc: dict) -> list[dict]:
    """Match page 1-2 spans near 'Securities registered' through exchange info."""
    try:
        texts = doc.get("texts", [])
        result = []
        in_section = False

        for i, s in enumerate(texts):
            page = s.get("page_no", 0)
            if page > 2:
                continue

            text = s.get("text", "").lower()

            # Start capturing at "Securities registered pursuant to Section 12(b)"
            if "securities registered" in text and "12(b)" in text:
                in_section = True

            if in_section:
                result.append(s)
                # Stop at next section indicator
                if "12(g)" in text or "indicate by check mark" in text.lower():
                    break

        return result
    except Exception:
        return []

def rule_page1_trading_exchange_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans with trading symbol or exchange keywords."""
    try:
        keywords = ["trading symbol", "title of each class", "name of each exchange",
                    "nasdaq", "new york stock exchange", "nyse", "stock exchange"]
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            any(kw in s.get("text", "").lower() for kw in keywords)
        ]
    except Exception:
        return []


def rule_securities_section_with_symbol(doc: dict) -> list[dict]:
    """Match Section 12(b) info AND spans mentioning trades under symbol."""
    try:
        texts = doc.get('texts', [])
        result = []
        seen = set()

        # Part 1: Section 12(b) on page 1-2
        in_section = False
        for i, s in enumerate(texts):
            page = s.get('page_no', 0)
            if page > 2:
                continue
            text = s.get('text', '').lower()
            if 'securities registered' in text and '12(b)' in text:
                in_section = True
            if in_section:
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))
                if '12(g)' in text or 'indicate by check mark' in text:
                    break

        # Part 2: Spans mentioning trading under symbol (for older filings)
        for s in texts:
            text = s.get('text', '').lower()
            if ('trades under' in text and 'symbol' in text) or \
               ('traded on' in text and 'symbol' in text):
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))

        return result
    except Exception:
        return []


if __name__ == "__main__":
    print("Ground truth loaded:")
    gt = load_ground_truth()
    for doc, ans in gt.items():
        print(f"  {doc}: {ans}")
    print()

    print("\n=== Testing rule_page1_trading_symbol_table ===")
    r1 = evaluate_rule(rule_page1_trading_symbol_table, verbose=True)
    print(f"Hit rate: {r1['hit_rate']:.2%}, Avg cost: {r1['avg_cost']:.4f}")

    print("\n=== Testing rule_page1_securities_section ===")
    r2 = evaluate_rule(rule_page1_securities_section, verbose=True)
    print(f"Hit rate: {r2['hit_rate']:.2%}, Avg cost: {r2['avg_cost']:.4f}")

    print("\n=== Testing rule_page1_trading_exchange_keywords ===")
    r3 = evaluate_rule(rule_page1_trading_exchange_keywords, verbose=True)
    print(f"Hit rate: {r3['hit_rate']:.2%}, Avg cost: {r3['avg_cost']:.4f}")
