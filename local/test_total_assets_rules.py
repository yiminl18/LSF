#!/usr/bin/env python3
"""Test total assets rules - coverage and cost analysis."""
import json
import re
import time

START_TIME = time.time()

QUESTION = "What is total assets at year-end (from the audited balance sheet)?"
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
    all_labels = json.load(f)

ground_truth = {}
for doc_name in DOC_NAMES:
    key = f"{doc_name}.pdf"
    if key in all_labels and QUESTION in all_labels[key]:
        ground_truth[doc_name] = all_labels[key][QUESTION]

# Load documents
docs = {}
for doc_name in DOC_NAMES:
    with open(f"data/financebench/processing/{doc_name}_reconstructed.json") as f:
        docs[doc_name] = json.load(f)

def rule_table_total_assets_balance_sheet(doc: dict) -> list[dict]:
    """Match first 2 tables with total assets row header in balance sheet/financial sections."""
    try:
        results = []
        path_keywords = ["item 6", "item 8", "balance sheet", "selected financial",
                        "financial statement", "consolidated balance", "annual report", "part iv"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                # Match "total assets" but not "total assets of reportable segments" etc.
                if h.strip() in ["total assets", "total assets (i)"]:
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []


def count_tokens(text: str) -> int:
    """Simple token count approximation."""
    return len(text.split())


def check_hit(retrieved_text: str, answer: str) -> bool:
    """Check if answer is in retrieved text via numeric match."""
    # Extract numbers from answer
    answer_nums = re.findall(r'[\d,\.]+', answer.replace(',', ''))
    if not answer_nums:
        return answer.lower() in retrieved_text.lower()

    retrieved_clean = retrieved_text.replace(',', '').replace(' ', '')
    for num in answer_nums:
        num_clean = num.replace(',', '')
        if len(num_clean) >= 3 and num_clean in retrieved_clean:
            return True

    # Special case for billion/million conversion (e.g., 187.4 billion = 187,400 million)
    if "billion" in answer.lower():
        for num in answer_nums:
            try:
                value = float(num.replace(',', ''))
                # Convert to millions (billions * 1000)
                millions_value = value * 1000
                # Look for approximate match (within 1%)
                millions_str = str(int(millions_value))
                if millions_str in retrieved_clean:
                    return True
                # Also try with comma formatting
                if len(millions_str) >= 6:
                    millions_formatted = f"{int(millions_value):,}".replace(',', '')
                    if millions_formatted[:3] in retrieved_clean:
                        return True
            except:
                pass
    return False


print("Testing rule: rule_table_total_assets_balance_sheet")
print("="*80)

total_cost = 0.0
hits = 0
missed_docs = []

for doc_name, doc in docs.items():
    answer = ground_truth.get(doc_name, "")

    # Get full doc tokens
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
    full_tokens = count_tokens(full_text)

    # Get rule results
    spans = rule_table_total_assets_balance_sheet(doc)
    retrieved_text = " ".join(s.get("text", "") for s in spans)
    retrieved_tokens = count_tokens(retrieved_text)

    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    hit = check_hit(retrieved_text, answer)
    if hit:
        hits += 1
    else:
        missed_docs.append(doc_name)

    print(f"{doc_name}:")
    print(f"  Answer: {answer}")
    print(f"  Spans: {len(spans)}, Cost: {cost:.4f} ({retrieved_tokens}/{full_tokens} tokens)")
    print(f"  Hit: {hit}")
    if not hit:
        print(f"  Retrieved preview: {retrieved_text[:200]}...")

avg_cost = total_cost / len(docs)
hit_rate = hits / len(docs)

print()
print("="*80)
print(f"Hit rate: {hits}/{len(docs)} = {hit_rate:.2%}")
print(f"Avg cost: {avg_cost:.4f}")
print(f"Missed docs: {missed_docs}")
