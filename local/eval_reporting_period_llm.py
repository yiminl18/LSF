#!/usr/bin/env python3
"""Evaluate reporting period rules using LLM judge."""

import json
import os
import sys
import time
import re
import importlib.util
from pathlib import Path
import tiktoken

# Add src to path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# Import model
from models import gpt54

# Document mapping
DOC_MAP = {
    "BOEING_2019_10K": "BOEING_2019_10K",
    "ADOBE_2020_10K": "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K": "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K": "COSTCO_2018_10K",
    "AMCOR_2019_10K": "AMCOR_2019_10K",
    "AMAZON_2020_10K": "AMAZON_2020_10K",
    "AMAZON_2019_10K": "AMAZON_2019_10K",
    "ADOBE_2021_10K": "ADOBE_2021_10K",
    "EBAY_2022_10K": "EBAY_2022_10K",
    "ADOBE_2019_10K": "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q": "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q": "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q": None,
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q": "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01": "AMCOR_2022_8K_dated-2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09": "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16": "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01": None,
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "FOOTLOCKER_2022_8K_dated-2022-05-20",
}

QUESTION = "What is the reporting period covered by this document (e.g. fiscal year ended, quarter ended, or event date)?"
QUESTION_SLUG = "what_is_the_reporting_period_covered_by_this_document__e_g"

enc = tiktoken.get_encoding("cl100k_base")

# Track LLM calls and tokens
total_llm_calls = 0
total_input_tokens = 0
total_output_tokens = 0

def load_doc(doc_name):
    actual_name = DOC_MAP.get(doc_name, doc_name)
    if actual_name is None:
        return None
    path = f"data/financebench/processing/{actual_name}_reconstructed.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_labels():
    with open("data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    gt = {}
    for doc_name in DOC_MAP.keys():
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            gt[doc_name] = labels[key][QUESTION]
    return gt

def load_rule(rule_path):
    spec = importlib.util.spec_from_file_location("_rule", str(rule_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if name.startswith("rule_") and callable(obj):
            return obj
    raise ValueError(f"No rule function found in {rule_path}")

def spans_to_context(spans):
    """Convert spans to context string."""
    parts = []
    for s in spans:
        text = s.get("text", "")
        text_span = s.get("text_span", "")
        if text:
            parts.append(text)
        if text_span:
            parts.append(text_span)
    return "\n".join(parts)

def call_llm_qa(question, context):
    """Call LLM to answer the question."""
    global total_llm_calls, total_input_tokens, total_output_tokens

    total_llm_calls += 1
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    input_tokens = len(enc.encode(prompt))
    total_input_tokens += input_tokens

    answer = gpt54.gpt_54(question, context)
    output_tokens = len(enc.encode(answer))
    total_output_tokens += output_tokens

    return answer

def call_llm_judge(question, predicted, ground_truth):
    """Call LLM judge to compare answers."""
    global total_llm_calls, total_input_tokens, total_output_tokens

    total_llm_calls += 1

    system = """You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "Fiscal year ended December 31, 2019" and "fiscal year ended December 31, 2019" as the same
- Treat "Quarterly period ended June 30, 2023" and "Quarter ended June 30, 2023" as the same
- For dates, treat "June 30, 2022" and "Fiscal year ended June 30, 2022" as the same if they represent the same period end
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    prompt = f"""Question: {question}

Predicted answer: {predicted}

Ground truth answer: {ground_truth}

Is the predicted answer correct?"""

    input_tokens = len(enc.encode(system + prompt))
    total_input_tokens += input_tokens

    result = gpt54.chat_completions(prompt, system=system, max_completion_tokens=10)
    output_tokens = len(enc.encode(result))
    total_output_tokens += output_tokens

    return "CORRECT" in result.upper()

def get_full_doc_tokens(doc):
    all_text = " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in doc.get("texts", []))
    return len(enc.encode(all_text))

def get_retrieved_tokens(spans):
    all_text = " ".join(s.get("text", "") + " " + s.get("text_span", "") for s in spans)
    return len(enc.encode(all_text))

def main():
    global total_llm_calls, total_input_tokens, total_output_tokens

    start_time = time.time()

    # Load data
    ground_truth = load_labels()
    docs = {}
    for doc_name in DOC_MAP.keys():
        doc = load_doc(doc_name)
        if doc:
            docs[doc_name] = doc

    print(f"Loaded {len(docs)} documents, {len(ground_truth)} ground truth answers")

    # Load rule
    rule_path = Path("rules/agent/financebench_mix_doc_claude") / QUESTION_SLUG / "rule_page1_period_keywords.py"
    rule_fn = load_rule(rule_path)
    print(f"Loaded rule from {rule_path}")

    # Evaluate
    results = []
    correct = 0
    total = 0
    costs = []

    for doc_name, gt_answer in ground_truth.items():
        doc = docs.get(doc_name)
        if doc is None:
            print(f"Skipping {doc_name} (not found)")
            continue

        total += 1
        print(f"\nEvaluating {doc_name}...")

        # Apply rule
        spans = rule_fn(doc)
        context = spans_to_context(spans)

        # Calculate cost
        full_tokens = get_full_doc_tokens(doc)
        retrieved_tokens = get_retrieved_tokens(spans)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        costs.append(cost)

        # Call LLM to answer
        predicted = call_llm_qa(QUESTION, context)
        print(f"  Ground truth: {gt_answer}")
        print(f"  Predicted: {predicted[:100]}...")

        # Call judge
        is_correct = call_llm_judge(QUESTION, predicted, gt_answer)
        if is_correct:
            correct += 1
            print(f"  Result: CORRECT")
        else:
            print(f"  Result: INCORRECT")

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt_answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "num_spans": len(spans),
            "cost": cost,
        })

    end_time = time.time()
    latency = end_time - start_time

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    accuracy = correct / total if total > 0 else 0
    avg_cost = sum(costs) / len(costs) if costs else 0

    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Avg cost: {avg_cost:.4f} ({avg_cost*100:.2f}%)")
    print(f"Latency: {latency:.2f}s")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

    # Save results
    output = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "latency": latency,
        "total_llm_calls": total_llm_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": results,
    }

    output_path = f"local/eval_reporting_period_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return accuracy, avg_cost, latency

if __name__ == "__main__":
    main()
