#!/usr/bin/env python3
"""Finalize total assets rules - compute metrics and save log file."""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

START_TIME = time.time()

QUESTION = "What is total assets at year-end (from the audited balance sheet)?"
QUESTION_SLUG = "what_is_total_assets_at_year_end__from_the_audited_balance_s"
MODEL = "claude-opus-4-5"

GROUND_TRUTH = {
    "AMCOR_2019_10K": "17,165.0 million",
    "COSTCO_2017_10K": "36,347",
    "BOEING_2018_10K": "$117,359 million",
    "AMAZON_2018_10K": "$162,648 million",
    "EBAY_2021_10K": "$26,626 million",
    "AMAZON_2016_10K": "$83,402 million",
    "CORNING_2022_10K": "29,499 million",
    "NIKE_2021_10K": "$37,740 million",
    "LOCKHEEDMARTIN_2022_10K": "$52,880 million",
    "JOHNSON_JOHNSON_2022_10K": "$187.4 billion",
}

DOC_NAMES = list(GROUND_TRUTH.keys())


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


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
                if h.strip() in ["total assets", "total assets (i)"]:
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []


def check_hit(retrieved_text: str, answer: str) -> bool:
    """Check if answer is in retrieved text via numeric match."""
    answer_nums = re.findall(r'[\d,\.]+', answer.replace(',', ''))
    if not answer_nums:
        return answer.lower() in retrieved_text.lower()

    retrieved_clean = retrieved_text.replace(',', '').replace(' ', '')
    for num in answer_nums:
        num_clean = num.replace(',', '')
        if len(num_clean) >= 3 and num_clean in retrieved_clean:
            return True

    # Special case for billion/million conversion
    # e.g., "$187.4 billion" should match "187,378" (in millions)
    if "billion" in answer.lower():
        for num in answer_nums:
            try:
                value = float(num.replace(',', ''))
                # Convert billions to millions
                millions_value = value * 1000
                # Check for approximate match (within 1% tolerance)
                # Look for numbers in retrieved text
                retrieved_nums = re.findall(r'[\d,]+', retrieved_text)
                for rnum in retrieved_nums:
                    try:
                        rval = float(rnum.replace(',', ''))
                        if abs(rval - millions_value) / millions_value < 0.01:
                            return True
                    except:
                        continue
            except:
                pass
    return False


def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)


print("="*80)
print(f"FINALIZING: {QUESTION}")
print("="*80)

# Compute metrics
total_cost = 0.0
hits = 0
coverage = 0

for doc_name in DOC_NAMES:
    doc = load_doc(doc_name)
    answer = GROUND_TRUTH[doc_name]

    # Get full doc tokens
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
    full_tokens = count_tokens(full_text)

    # Apply rule
    spans = rule_table_total_assets_balance_sheet(doc)
    retrieved_text = "\n\n".join(s.get("text", "") for s in spans)
    retrieved_tokens = count_tokens(retrieved_text)

    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    hit = check_hit(retrieved_text, answer)
    if hit:
        hits += 1
    if len(spans) > 0:
        coverage += 1

    status = "✓" if hit else "✗"
    print(f"{status} {doc_name}: cost={cost:.4f}, spans={len(spans)}")

avg_cost = total_cost / len(DOC_NAMES)
hit_rate = hits / len(DOC_NAMES)

print()
print("="*80)
print("METRICS")
print("="*80)
print(f"Hit rate (substring match): {hits}/{len(DOC_NAMES)} = {hit_rate:.2%}")
print(f"Avg cost ratio: {avg_cost:.4f}")
print(f"Coverage: {coverage}/{len(DOC_NAMES)}")

latency = time.time() - START_TIME
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
iso_timestamp = datetime.now(timezone.utc).isoformat()

# Create log file
log = {
    "question": QUESTION,
    "question_slug": QUESTION_SLUG,
    "timestamp": iso_timestamp,
    "latency_seconds": round(latency, 2),
    "agent_input_tokens": 0,  # Not tracked - no LLM API access in this session
    "agent_output_tokens": 0,
    "total_llm_calls": 0,
    "num_rules": 1,
    "merge_accuracy": hit_rate,  # Using substring match as proxy
    "avg_cost_ratio": round(avg_cost, 6),
    "rules": [
        {
            "rule_name": "rule_table_total_assets_balance_sheet",
            "description": "Match first 2 tables with total assets row header in balance sheet/financial sections.",
            "coverage": coverage,
            "avg_cost_ratio": round(avg_cost, 6),
            "file": f"rules/agent/financebench_agent/{QUESTION_SLUG}/rule_table_total_assets_balance_sheet.py"
        }
    ],
    "note": "merge_accuracy based on substring match proxy; LLM judge evaluation pending API access"
}

log_path = Path(f"rules/agent/financebench_agent/{QUESTION_SLUG}_{MODEL}_{timestamp}_rule_gen.json")
log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
print(f"\nLog saved to: {log_path}")

print()
print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Number of rules: 1")
print(f"Merge accuracy (proxy): {hit_rate:.2%}")
print(f"Avg cost ratio: {avg_cost:.4f}")
print(f"Latency: {latency:.2f}s")
print()
print("Rule: rule_table_total_assets_balance_sheet")
print(f"  Coverage: {coverage}/{len(DOC_NAMES)} docs")
print(f"  Avg cost: {avg_cost:.4f}")
