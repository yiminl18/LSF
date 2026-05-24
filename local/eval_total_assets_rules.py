#!/usr/bin/env python3
"""Evaluate total assets rules using Anthropic Claude API."""

import json
import time
from pathlib import Path

import anthropic

START_TIME = time.time()

QUESTION = "What is total assets at year-end (from the audited balance sheet)?"
QUESTION_SLUG = "what_is_total_assets_at_year_end__from_the_audited_balance_s"

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

# Token counters
total_input_tokens = 0
total_output_tokens = 0
total_llm_calls = 0


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


def load_doc(doc_name):
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)


def get_llm_answer(client, retrieved_text, question):
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    total_llm_calls += 1
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    return response.content[0].text.strip()


def judge_answer(client, question, ground_truth, predicted):
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = f"Question: {question}\nGround Truth: {ground_truth}\nPredicted: {predicted}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    total_llm_calls += 1
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    return response.content[0].text.strip().upper() == "CORRECT"


print("="*80)
print(f"Evaluating: {QUESTION}")
print("="*80)

client = anthropic.Anthropic()
results = []
total_cost = 0.0
correct_count = 0

for doc_name in DOC_NAMES:
    doc = load_doc(doc_name)
    ground_truth = GROUND_TRUTH[doc_name]

    # Get full doc tokens
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
    full_tokens = count_tokens(full_text)

    # Apply rule
    spans = rule_table_total_assets_balance_sheet(doc)
    retrieved_text = "\n\n".join(s.get("text", "") for s in spans)
    retrieved_tokens = count_tokens(retrieved_text)

    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    # Get LLM answer
    predicted_answer = get_llm_answer(client, retrieved_text, QUESTION)

    # Judge answer
    correct = judge_answer(client, QUESTION, ground_truth, predicted_answer)
    if correct:
        correct_count += 1

    results.append({
        "doc_name": doc_name,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "correct": correct,
        "cost": cost,
        "retrieved_tokens": retrieved_tokens,
        "total_tokens": full_tokens,
    })

    status = "✓" if correct else "✗"
    print(f"{status} {doc_name}:")
    print(f"    Ground truth: {ground_truth}")
    print(f"    Predicted:    {predicted_answer}")
    print(f"    Cost:         {cost:.4f}")

avg_cost = total_cost / len(DOC_NAMES)
merge_accuracy = correct_count / len(DOC_NAMES)
latency = time.time() - START_TIME

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Merge accuracy: {correct_count}/{len(DOC_NAMES)} = {merge_accuracy:.2%}")
print(f"Avg cost ratio: {avg_cost:.4f}")
print(f"Total LLM calls: {total_llm_calls}")
print(f"Total input tokens: {total_input_tokens}")
print(f"Total output tokens: {total_output_tokens}")
print(f"Latency: {latency:.2f}s")

# Find failed docs
failed = [r for r in results if not r["correct"]]
if failed:
    print()
    print("Failed documents:")
    for r in failed:
        print(f"  - {r['doc_name']}: GT={r['ground_truth']}, Pred={r['predicted_answer']}")

# Save results
output = {
    "question": QUESTION,
    "question_slug": QUESTION_SLUG,
    "merge_accuracy": merge_accuracy,
    "avg_cost_ratio": avg_cost,
    "total_llm_calls": total_llm_calls,
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "latency_seconds": latency,
    "results": results,
}

output_path = Path("local/eval_total_assets_results.json")
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nResults saved to {output_path}")
