#!/usr/bin/env python3
"""Evaluate revenue rules with LLM judge using Azure OpenAI API."""

import json
import sys
import time
from pathlib import Path

# Add src to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tiktoken
from models import gpt54

# Configuration
DOC_NAMES = [
    "AMCOR_2019_10K", "COSTCO_2017_10K", "BOEING_2018_10K", "AMAZON_2018_10K",
    "EBAY_2021_10K", "AMAZON_2016_10K", "CORNING_2022_10K", "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K", "JOHNSON_JOHNSON_2022_10K"
]

GROUND_TRUTH = {
    "AMCOR_2019_10K": "9,458.2 million",
    "COSTCO_2017_10K": "126,172",
    "BOEING_2018_10K": "$101,127 million",
    "AMAZON_2018_10K": "$232,887 million",
    "EBAY_2021_10K": "$10,420 million",
    "AMAZON_2016_10K": "$135,987 million",
    "CORNING_2022_10K": "14,189 million",
    "NIKE_2021_10K": "$44,538 million",
    "LOCKHEEDMARTIN_2022_10K": "$65,984 million",
    "JOHNSON_JOHNSON_2022_10K": "$94.9 billion"
}

QUESTION = "What is total revenue for the most recent fiscal year (from the audited income statement)?"

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
total_llm_calls = 0

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def load_doc(doc_name):
    path = _ROOT / f"data/financebench/processing/{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

def rule_table_revenue_financial_sections(doc):
    """Match first 2 tables with revenue/sales row headers in financial sections."""
    try:
        results = []
        revenue_keywords = ["revenue", "net sales", "total sales", "sales to customer"]
        path_keywords = ["item 6", "item 7", "item 8", "selected financial",
                        "management's discussion", "financial statement",
                        "consolidated statement", "statement of income",
                        "statement of earnings", "statement of operations"]
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path_text = span.get("structure", {}).get("path_text", "").lower()
            if not any(kw in path_text for kw in path_keywords):
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_headers = [c.get("text", "").lower() for c in cells if c.get("is_row_header")]
            for h in row_headers:
                if any(kw in h for kw in revenue_keywords):
                    results.append(span)
                    break
            if len(results) >= 2:
                break
        return results
    except Exception:
        return []

def call_qa_llm(retrieved_text, question):
    """Call LLM to answer question based on retrieved text."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = """You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply with "NOT FOUND".
Return only the answer — a short value or phrase, not a full sentence."""

    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=200,
        temperature=0.0
    )

    usage = response.usage
    total_input_tokens += usage.prompt_tokens if usage else 0
    total_output_tokens += usage.completion_tokens if usage else 0
    total_llm_calls += 1

    return (response.choices[0].message.content or "").strip()

def call_judge_llm(question, ground_truth, predicted):
    """Call LLM to judge if predicted answer is correct."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = """You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect
- Numbers should match (e.g., "126,172" and "$126,172" are equivalent)
- "94.9 billion" and "94,943 million" are equivalent (same numeric value)

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = f"""Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}"""

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=10,
        temperature=0.0
    )

    usage = response.usage
    total_input_tokens += usage.prompt_tokens if usage else 0
    total_output_tokens += usage.completion_tokens if usage else 0
    total_llm_calls += 1

    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict == "CORRECT"

def main():
    global total_input_tokens, total_output_tokens, total_llm_calls

    print("=== Evaluating Total Revenue Rules with LLM Judge ===\n")

    results = []
    correct_count = 0
    total_cost = 0

    for doc_name in DOC_NAMES:
        print(f"Processing {doc_name}...")

        # Load document
        doc = load_doc(doc_name)
        full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
        full_tokens = count_tokens(full_text)

        # Apply rule
        spans = rule_table_revenue_financial_sections(doc)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)
        retrieved_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        # Get LLM answer
        if retrieved_text:
            predicted = call_qa_llm(retrieved_text, QUESTION)
        else:
            predicted = "NOT FOUND"

        # Judge
        gt = GROUND_TRUTH[doc_name]
        correct = call_judge_llm(QUESTION, gt, predicted)

        if correct:
            correct_count += 1
            status = "CORRECT"
        else:
            status = "INCORRECT"

        print(f"  GT: {gt}")
        print(f"  Predicted: {predicted}")
        print(f"  Status: {status}, Cost: {cost:.4f}")

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": correct,
            "cost": cost,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens
        })

    # Calculate metrics
    merge_accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)

    print(f"\n=== Summary ===")
    print(f"Merge Accuracy: {merge_accuracy:.2%} ({correct_count}/{len(DOC_NAMES)})")
    print(f"Avg Cost Ratio: {avg_cost:.4f}")
    print(f"Total LLM Calls: {total_llm_calls}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")

    # Show incorrect cases
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print(f"\n=== Incorrect Cases ===")
        for r in incorrect:
            print(f"  {r['doc_name']}: GT={r['ground_truth']}, Pred={r['predicted']}")

    return {
        "merge_accuracy": merge_accuracy,
        "avg_cost": avg_cost,
        "results": results,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls
    }

if __name__ == "__main__":
    result = main()

    # Save results to JSON
    output = {
        "question": QUESTION,
        "merge_accuracy": result["merge_accuracy"],
        "avg_cost_ratio": result["avg_cost"],
        "total_llm_calls": result["total_llm_calls"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "per_document": result["results"]
    }

    output_path = _ROOT / "local" / "eval_revenue_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")
