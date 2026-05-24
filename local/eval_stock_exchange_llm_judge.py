#!/usr/bin/env python3
"""LLM Judge evaluation for stock exchange rule."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import gpt54

GROUND_TRUTH = {
    "BOEING_2019_10K": "New York Stock Exchange",
    "ADOBE_2020_10K": "NASDAQ",
    "ACTIVISIONBLIZZARD_2020_10K": "The Nasdaq Global Select Market",
    "COSTCO_2018_10K": "The NASDAQ Global Select Market",
    "AMCOR_2019_10K": "The New York Stock Exchange",
    "AMAZON_2020_10K": "Nasdaq Global Select Market",
    "AMAZON_2019_10K": "Nasdaq Global Select Market",
    "ADOBE_2021_10K": "NASDAQ",
    "EBAY_2022_10K": "The Nasdaq Global Select Market",
    "ADOBE_2019_10K": "NASDAQ",
    "AMCOR_2023Q2_10Q": "New York Stock Exchange",
    "ADOBE_2022Q2_10Q": "NASDAQ",
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "The Nasdaq Global Select Market",
    "3M_2023Q2_10Q": "New York Stock Exchange",
    "AMCOR_2022_8K_2022-07-01": "New York Stock Exchange",
    "COSTCO_2023_8K_dated-2023-08-09": "NASDAQ",
    "COSTCO_2023_8K_dated-2023-08-16": "NASDAQ",
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "New York Stock Exchange"
}

QUESTION = "What stock exchange is the company's primary common stock listed on?"

def rule_page1_exchange_keywords(doc: dict) -> list[dict]:
    """Match page 1 spans containing stock exchange keywords."""
    try:
        keywords = ["nasdaq", "new york stock exchange", "stock exchange"]
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and
            any(kw in s.get("text", "").lower() for kw in keywords)
        ]
    except Exception:
        return []

def get_answer_from_llm(retrieved_text: str) -> tuple[str, dict]:
    """Call LLM to answer the question based on retrieved text."""
    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Passage:\n{retrieved_text}\n\nQuestion: {QUESTION}"},
        ],
        max_completion_tokens=100,
        temperature=0.0,
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
    return answer, usage

def judge_answer(predicted: str, ground_truth: str) -> tuple[bool, dict]:
    """Use LLM to judge if predicted answer matches ground truth."""
    judge_system = """\
You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "NYSE" and "New York Stock Exchange" as the same
- Treat "NASDAQ" and "The NASDAQ Global Select Market" as equivalent (both refer to NASDAQ)
- Treat "Nasdaq Global Select Market" and "The Nasdaq Global Select Market" as equivalent
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": f"Question: {QUESTION}\nPredicted answer: {predicted}\nGround truth: {ground_truth}"},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )

    verdict = (response.choices[0].message.content or "").strip().upper()
    is_correct = "CORRECT" in verdict
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }
    return is_correct, usage

def main():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    correct = 0
    total_cost = 0

    for doc_name, gt_answer in GROUND_TRUTH.items():
        print(f"Processing {doc_name}...")

        # Load document
        with open(f"data/financebench/processing/{doc_name}_reconstructed.json") as f:
            doc = json.load(f)

        # Apply rule
        spans = rule_page1_exchange_keywords(doc)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        # Calculate cost
        full_text = " ".join(s.get("text", "") for s in doc["texts"])
        full_tokens = len(enc.encode(full_text))
        span_tokens = len(enc.encode(retrieved_text))
        cost = span_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        # Get LLM answer
        predicted, qa_usage = get_answer_from_llm(retrieved_text)
        total_input_tokens += qa_usage["input_tokens"]
        total_output_tokens += qa_usage["output_tokens"]
        total_llm_calls += 1

        # Judge answer
        is_correct, judge_usage = judge_answer(predicted, gt_answer)
        total_input_tokens += judge_usage["input_tokens"]
        total_output_tokens += judge_usage["output_tokens"]
        total_llm_calls += 1

        if is_correct:
            correct += 1

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt_answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "cost": cost,
            "num_spans": len(spans)
        })

        print(f"  Predicted: {predicted}")
        print(f"  Ground truth: {gt_answer}")
        print(f"  Correct: {is_correct}")

    merge_accuracy = correct / len(results)
    avg_cost = total_cost / len(results)

    print("\n" + "="*60)
    print(f"Merge Accuracy: {merge_accuracy:.2%} ({correct}/{len(results)})")
    print(f"Avg Cost: {avg_cost:.5f}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    # Show failures
    failures = [r for r in results if not r["is_correct"]]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  {f['doc_name']}: predicted '{f['predicted']}', expected '{f['ground_truth']}'")

    return {
        "merge_accuracy": merge_accuracy,
        "avg_cost": avg_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "results": results
    }

if __name__ == "__main__":
    main()
