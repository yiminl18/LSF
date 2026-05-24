#!/usr/bin/env python3
"""Test script for address of principal executive offices rule."""

import json
import tiktoken

# Ground truth
docs = ['AMCOR_2019_10K', 'COSTCO_2017_10K', 'BOEING_2018_10K', 'AMAZON_2018_10K', 'EBAY_2021_10K',
        'AMAZON_2016_10K', 'CORNING_2022_10K', 'NIKE_2021_10K', 'LOCKHEEDMARTIN_2022_10K', 'JOHNSON_JOHNSON_2022_10K']

question = 'What is the address of principal executive offices and ZIP code?'

# Load labels
with open('data/financebench/sample_doc_labels.json') as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in docs:
    ground_truth[doc_name] = labels[f'{doc_name}.pdf'][question]


def rule_page1_address_principal(doc: dict) -> list[dict]:
    """Match page 1 spans containing or near 'address of principal executive offices'."""
    try:
        results = []
        texts = doc.get("texts", [])
        seen_ids = set()

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue

            text = span.get("text", "").lower()
            text_span = span.get("text_span", "").lower()

            # Pattern 1: text_span contains the address label - span.text has the address
            if "address of principal" in text_span or "address and telephone" in text_span:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))

            # Pattern 2: text contains the address label - previous span has address
            if "address of principal" in text or "address and telephone" in text:
                # Add the label span itself (in case it has text_span with address)
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                # Add the previous span(s) which likely contain the address
                for offset in [1, 2, 3]:
                    prev_idx = i - offset
                    if prev_idx >= 0 and texts[prev_idx].get("page_no") == 1:
                        prev = texts[prev_idx]
                        if id(prev) not in seen_ids:
                            # Skip if it's another label span
                            prev_text = prev.get("text", "").lower()
                            if "jurisdiction" in prev_text or "i.r.s." in prev_text or "employer identification" in prev_text:
                                continue
                            results.append(prev)
                            seen_ids.add(id(prev))

            # Pattern 3: Also capture "(zip code)" patterns
            if "(zip code)" in text.lower():
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                # Previous span may have the zip
                if i > 0 and texts[i-1].get("page_no") == 1:
                    prev = texts[i-1]
                    if id(prev) not in seen_ids:
                        results.append(prev)
                        seen_ids.add(id(prev))

        return results
    except Exception:
        return []


# Test the rule
enc = tiktoken.encoding_for_model("gpt-4")

total_cost = 0
hits = 0

print("Testing rule_page1_address_principal\n")

for doc_name in docs:
    with open(f'data/financebench/processing/{doc_name}_reconstructed.json') as f:
        doc = json.load(f)

    # Apply rule
    retrieved = rule_page1_address_principal(doc)

    # Get retrieved text
    retrieved_text = " ".join([s.get("text", "") + " " + s.get("text_span", "") for s in retrieved])

    # Get full document text
    full_text = " ".join([s.get("text", "") + " " + s.get("text_span", "") for s in doc["texts"]])

    # Calculate cost
    retrieved_tokens = len(enc.encode(retrieved_text))
    full_tokens = len(enc.encode(full_text))
    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    # Check hit (substring match)
    gt = ground_truth[doc_name]
    # Normalize for comparison
    gt_norm = gt.lower().replace(",", "").replace(".", "")
    retrieved_norm = retrieved_text.lower().replace(",", "").replace(".", "")

    # Check if key parts of gt are in retrieved text
    gt_parts = gt_norm.split()
    hit = all(part in retrieved_norm for part in gt_parts[:3])  # First 3 words
    if hit:
        hits += 1

    print(f"{doc_name}")
    print(f"  GT: {gt}")
    print(f"  Retrieved spans: {len(retrieved)}")
    print(f"  Cost: {cost:.4f} ({retrieved_tokens}/{full_tokens} tokens)")
    print(f"  Hit: {'YES' if hit else 'NO'}")
    if not hit:
        print(f"  Retrieved text: {retrieved_text[:200]}...")
    print()

print("=" * 60)
print(f"Substring match accuracy: {hits}/{len(docs)} = {hits/len(docs):.2%}")
print(f"Average cost: {total_cost/len(docs):.4f}")
