#!/usr/bin/env python3
"""Evaluate telephone number rules using Anthropic Claude API."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import tiktoken

QUESTION = "What is the registrant's telephone number?"
QUESTION_SLUG = "what_is_the_registrant_s_telephone_number"

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

DOC_NAMES = list(GROUND_TRUTH.keys())
DATA_DIR = Path("data/financebench/processing")

def count_tokens(text):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        return len(text) // 4

def load_doc(doc_name):
    path = DATA_DIR / f"{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

# === RULE ===

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

def get_llm_answer(client, retrieved_text, question):
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

    usage = response.usage
    return {
        "answer": response.content[0].text.strip() if response.content else "",
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
    }

def judge_answer(client, question, predicted, ground_truth):
    system_prompt = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Phone numbers with different formatting but same digits are equivalent
- (425) 313-8100 and 425-313-8100 are the same
- +44 117 9753200 and +44-117-9753200 are the same
- Ignore leading/trailing whitespace, punctuation differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    verdict = response.content[0].text.strip().lower() if response.content else ""
    usage = response.usage

    return {
        "correct": verdict == "correct",
        "verdict": verdict,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
    }

def main():
    print("Telephone Number Rules - LLM Judge Evaluation (Anthropic)")
    print("=" * 60)

    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    start_time = time.time()

    # Initialize Anthropic client
    client = anthropic.Anthropic()
    print("Using Anthropic Claude API")

    results = []
    correct_count = 0
    total_cost = 0

    for doc_name in DOC_NAMES:
        print(f"\n{doc_name}:")

        doc = load_doc(doc_name)
        spans = rule_page1_phone_pattern(doc)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        print(f"  Retrieved {len(spans)} spans, {retrieved_tokens} tokens, cost={cost:.4f}")

        qa_result = get_llm_answer(client, retrieved_text, QUESTION)
        predicted = qa_result["answer"]
        total_input_tokens += qa_result["input_tokens"]
        total_output_tokens += qa_result["output_tokens"]
        total_llm_calls += 1

        print(f"  Predicted: {predicted}")

        ground_truth = GROUND_TRUTH[doc_name]
        judge_result = judge_answer(client, QUESTION, predicted, ground_truth)
        total_input_tokens += judge_result["input_tokens"]
        total_output_tokens += judge_result["output_tokens"]
        total_llm_calls += 1

        if judge_result["correct"]:
            correct_count += 1
            print(f"  ✓ CORRECT (ground truth: {ground_truth})")
        else:
            print(f"  ✗ INCORRECT (ground truth: {ground_truth})")

        results.append({
            "doc_name": doc_name,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": judge_result["correct"],
            "cost_ratio": cost,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens,
        })

    elapsed = time.time() - start_time
    accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Merge Accuracy: {correct_count}/{len(DOC_NAMES)} = {accuracy:.2%}")
    print(f"Avg Cost Ratio: {avg_cost:.6f}")
    print(f"Total LLM Calls: {total_llm_calls}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Elapsed Time: {elapsed:.2f}s")

    # Save log file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(elapsed, 2),
        "agent_input_tokens": total_input_tokens,
        "agent_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "num_rules": 1,
        "merge_accuracy": round(accuracy, 4),
        "avg_cost_ratio": round(avg_cost, 6),
        "rules": [
            {
                "rule_name": "rule_page1_phone_pattern",
                "description": "Match page 1 spans containing phone number patterns.",
                "coverage": len(DOC_NAMES),
                "avg_cost_ratio": round(avg_cost, 6),
                "file": f"rules/agent/financebench_agent/{QUESTION_SLUG}/rule_page1_phone_pattern.py"
            }
        ],
        "per_document": results,
    }

    log_path = Path(f"rules/agent/financebench_agent/{QUESTION_SLUG}_claude-opus-4-5_{timestamp}_rule_gen.json")
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"\nLog saved to: {log_path}")

    return log

if __name__ == "__main__":
    main()
