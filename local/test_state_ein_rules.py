#!/usr/bin/env python3
"""Test state/EIN rules and measure cost/accuracy."""

import json
import os
import re
import time

START_TIME = time.time()

SAMPLED_DOCS = [
    "BOEING_2019_10K", "ADOBE_2020_10K", "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K", "AMCOR_2019_10K", "AMAZON_2020_10K", "AMAZON_2019_10K",
    "ADOBE_2021_10K", "EBAY_2022_10K", "ADOBE_2019_10K", "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q", "Pfizer_2023Q2_10Q", "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q", "AMCOR_2022_8K_2022-07-01", "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16", "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is the state (or other jurisdiction) of incorporation and the IRS Employer Identification Number?"

# Load ground truth
with open("data/financebench/sample_mix_doc_labels.json") as f:
    all_labels = json.load(f)

ground_truth = {}
for doc_name in SAMPLED_DOCS:
    pdf_name = doc_name + ".pdf"
    if pdf_name in all_labels and QUESTION in all_labels[pdf_name]:
        ground_truth[doc_name] = all_labels[pdf_name][QUESTION]

# Load docs
docs = {}
for doc_name in SAMPLED_DOCS:
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if os.path.exists(path):
        with open(path) as f:
            docs[doc_name] = json.load(f)

print(f"Loaded {len(docs)} docs, {len(ground_truth)} ground truths")

# Rule definitions
def rule_page1_state_ein_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans containing state/EIN labels or EIN pattern."""
    try:
        import re
        results = []
        ein_pattern = re.compile(r'\d{2}-\d{7}')

        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower()

            # Match jurisdiction/incorporation label
            if "jurisdiction" in text_lower and "incorporation" in text_lower:
                results.append(span)
            # Match employer identification label
            elif "employer identification" in text_lower or "i.r.s." in text_lower:
                results.append(span)
            # Match EIN pattern
            elif ein_pattern.search(text):
                results.append(span)

        return results
    except Exception:
        return []


def rule_page1_state_ein_context(doc: dict) -> list[dict]:
    """Match page 1 spans containing state/EIN values with context."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        ein_pattern = re.compile(r'\d{2}-\d{7}')

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower()

            # Match state of incorporation label - flexible patterns
            if "state or other jurisdiction" in text_lower:
                results.append(span)
                # Get previous span (the state value) if it's short
                if i > 0 and texts[i-1].get("page_no") == 1:
                    prev = texts[i-1]
                    if len(prev.get("text", "")) < 100:
                        results.append(prev)
            # Also match variations
            elif "jurisdiction" in text_lower and "incorporation" in text_lower:
                results.append(span)
                if i > 0 and texts[i-1].get("page_no") == 1:
                    prev = texts[i-1]
                    if len(prev.get("text", "")) < 100:
                        results.append(prev)

            # Match I.R.S. Employer label - multiple variations
            if "employer identification" in text_lower:
                results.append(span)
                # Get previous span for EIN value if not in same span
                if not ein_pattern.search(text) and i > 0 and texts[i-1].get("page_no") == 1:
                    results.append(texts[i-1])

            # Match EIN directly (pattern XX-XXXXXXX)
            if ein_pattern.search(text):
                results.append(span)

        return results
    except Exception:
        return []


def rule_page1_cover_page_info(doc: dict) -> list[dict]:
    """Match page 1 spans that could contain cover page registration info."""
    try:
        import re
        results = []
        ein_pattern = re.compile(r'\d{2}-\d{7}')

        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower()

            # Key cover page patterns
            if any(kw in text_lower for kw in [
                "state or other jurisdiction",
                "jurisdiction of incorporation",
                "employer identification",
                "i.r.s. employer",
                "irs employer",
                "(i.r.s.",
            ]):
                results.append(span)
            elif ein_pattern.search(text):
                results.append(span)

        return results
    except Exception:
        return []


def rule_page1_state_ein_complete(doc: dict) -> list[dict]:
    """Match page 1 spans containing state/EIN info with broader context."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        ein_pattern = re.compile(r'\d{2}-\d{7}')

        # Common jurisdictions
        jurisdictions = {
            "delaware", "washington", "california", "new york", "texas",
            "jersey", "nevada", "florida", "illinois", "massachusetts",
            "maryland", "pennsylvania", "ohio", "georgia", "north carolina",
            "virginia", "colorado", "arizona", "michigan", "minnesota",
        }

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower().strip()

            # Match state of incorporation label
            if "state or other jurisdiction" in text_lower:
                results.append(span)
                # Look back up to 3 spans for the state value
                for j in range(1, 4):
                    if i - j >= 0 and texts[i-j].get("page_no") == 1:
                        prev_text = texts[i-j].get("text", "").lower().strip()
                        # Check if it looks like a state name (short text matching known states)
                        if prev_text in jurisdictions:
                            results.append(texts[i-j])
                            break
                        # Or if it contains a known state name
                        for state in jurisdictions:
                            if state in prev_text and len(prev_text) < 50:
                                results.append(texts[i-j])
                                break

            # Match employer identification label
            if "employer identification" in text_lower:
                results.append(span)
                # Get previous span for EIN value if not in same span
                if not ein_pattern.search(text) and i > 0 and texts[i-1].get("page_no") == 1:
                    results.append(texts[i-1])

            # Match EIN directly
            if ein_pattern.search(text):
                results.append(span)

            # Match state names directly (short spans with just state name)
            if text_lower in jurisdictions:
                results.append(span)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in results:
            sid = id(s)
            if sid not in seen:
                seen.add(sid)
                unique.append(s)
        return unique
    except Exception:
        return []


# Test rules
def get_doc_text(doc):
    """Get full document text."""
    return " ".join(s.get("text", "") for s in doc.get("texts", []))


def get_span_text(spans):
    """Get concatenated text from spans."""
    return " ".join(s.get("text", "") for s in spans)


def tiktoken_count(text):
    """Approximate token count."""
    # Rough approximation: ~4 chars per token
    return len(text) // 4


def test_rule(rule_fn, name):
    """Test a rule and return metrics."""
    hits = 0
    total_cost = 0
    covered = []
    uncovered = []

    for doc_name in docs:
        if doc_name not in ground_truth:
            continue

        doc = docs[doc_name]
        gt = ground_truth[doc_name]

        # Apply rule
        spans = rule_fn(doc)
        retrieved_text = get_span_text(spans)
        full_text = get_doc_text(doc)

        # Check hit (substring match)
        # Parse ground truth - usually format like "Delaware, 91-0425694"
        parts = re.split(r'[,;]\s*', gt)
        state_part = parts[0].strip().lower() if parts else ""
        ein_part = parts[1].strip() if len(parts) > 1 else ""

        state_hit = state_part in retrieved_text.lower() if state_part else True
        ein_hit = ein_part in retrieved_text if ein_part else True

        if state_hit and ein_hit:
            hits += 1
            covered.append(doc_name)
        else:
            uncovered.append(doc_name)

        # Calculate cost
        if full_text:
            cost = tiktoken_count(retrieved_text) / tiktoken_count(full_text)
            total_cost += cost

    n_docs = len([d for d in docs if d in ground_truth])
    accuracy = hits / n_docs if n_docs > 0 else 0
    avg_cost = total_cost / n_docs if n_docs > 0 else 0

    print(f"\n{name}:")
    print(f"  Accuracy: {hits}/{n_docs} = {accuracy:.2%}")
    print(f"  Avg cost: {avg_cost:.4f}")
    print(f"  Covered: {covered}")
    print(f"  Uncovered: {uncovered}")

    return accuracy, avg_cost, covered, uncovered


# Run tests
print("\n=== RULE TESTS ===")
test_rule(rule_page1_state_ein_keywords, "rule_page1_state_ein_keywords")
test_rule(rule_page1_state_ein_context, "rule_page1_state_ein_context")
test_rule(rule_page1_cover_page_info, "rule_page1_cover_page_info")
test_rule(rule_page1_state_ein_complete, "rule_page1_state_ein_complete")

# Debug the best rule
print("\n=== VERIFY COMPLETE RULE ===")
for doc_name in ["FOOTLOCKER_2022_8K_dated-2022-05-20", "COSTCO_2023_8K_dated-2023-08-09", "AMCOR_2022_8K_2022-07-01"]:
    if doc_name not in docs:
        print(f"\n{doc_name}: NOT LOADED")
        continue
    doc = docs[doc_name]
    gt = ground_truth.get(doc_name, "N/A")

    # Apply the best rule
    spans = rule_page1_state_ein_complete(doc)
    retrieved_text = get_span_text(spans)

    print(f"\n{doc_name}:")
    print(f"  GT: {gt}")
    print(f"  Retrieved spans ({len(spans)}):")
    for s in spans:
        print(f"    - [{s.get('page_no')}] {s.get('text')[:80]}...")

    # Check what's matching
    parts = re.split(r'[,;]\s*', gt)
    state_part = parts[0].strip().lower() if parts else ""
    ein_part = parts[1].strip() if len(parts) > 1 else ""
    print(f"  State '{state_part}' in text: {state_part in retrieved_text.lower()}")
    print(f"  EIN '{ein_part}' in text: {ein_part in retrieved_text}")

# Show timing
print(f"\nTotal time: {time.time() - START_TIME:.2f}s")
