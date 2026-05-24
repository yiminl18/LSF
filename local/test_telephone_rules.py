#!/usr/bin/env python3
"""Test script for telephone rules."""
import json
import re
import tiktoken


def rule_page1_telephone_keyword(doc: dict) -> list[dict]:
    """Return page 1 spans containing the telephone label or phone number pattern."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        # Phone number patterns
        phone_pattern = re.compile(r'[\+\(]?\d{2,3}[\)\s\-]?\s*\d{3}[\s\-]?\d{4}')

        for span in texts:
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower()

            # Match if contains "telephone" keyword
            if "telephone" in text_lower:
                results.append(span)
            # Or if it contains a phone number pattern
            elif phone_pattern.search(text):
                results.append(span)

        return results
    except Exception:
        return []


# Load all documents and test
sampled_docs = [
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
    'ACTIVSIONBLIZZARD_2023Q2_10Q',
    '3M_2023Q2_10Q',
    'AMCOR_2022_8K_2022-07-01',
    'COSTCO_2023_8K_dated-2023-08-09',
    'COSTCO_2023_8K_dated-2023-08-16',
    'FOOTLOCKER_2022_8K_dated-2022-05-20'
]

# Load ground truth
with open('data/financebench/sample_mix_doc_labels.json') as f:
    labels = json.load(f)

question = "What is the registrant's telephone number?"

ground_truth = {}
for doc_name in sampled_docs:
    pdf_name = doc_name + '.pdf'
    if pdf_name in labels:
        answer = labels[pdf_name].get(question)
        if answer:
            ground_truth[doc_name] = answer

# Test rules
enc = tiktoken.get_encoding("cl100k_base")

hits = 0
total_cost = 0
for doc_name in sampled_docs:
    path = f'data/financebench/processing/{doc_name}_reconstructed.json'
    try:
        with open(path) as f:
            doc = json.load(f)
    except:
        continue

    answer = ground_truth.get(doc_name, '')
    answer_normalized = re.sub(r'[\s\-\(\)\+]', '', answer)

    # Apply rule
    retrieved = rule_page1_telephone_keyword(doc)

    # Get retrieved text
    retrieved_text = " ".join(s.get("text", "") for s in retrieved)
    retrieved_normalized = re.sub(r'[\s\-\(\)\+]', '', retrieved_text)

    # Check hit
    is_hit = answer_normalized in retrieved_normalized
    if is_hit:
        hits += 1

    # Calculate cost
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
    retrieved_tokens = len(enc.encode(retrieved_text))
    full_tokens = len(enc.encode(full_text))
    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    status = "✓" if is_hit else "✗"
    print(f"{status} {doc_name}: {len(retrieved)} spans, cost={cost:.4f}")
    if not is_hit:
        print(f"   Answer: {answer}")
        print(f"   Retrieved (first 200 chars): {retrieved_text[:200]}")

print(f"\nHit rate: {hits}/{len(sampled_docs)} = {hits/len(sampled_docs):.2%}")
print(f"Avg cost: {total_cost/len(sampled_docs):.4f}")
