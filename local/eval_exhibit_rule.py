#!/usr/bin/env python3
"""Evaluate exhibit index rule with LLM judge."""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tiktoken

# Document list
DOCS = [
    "BOEING_2019_10K", "ADOBE_2020_10K", "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K", "AMCOR_2019_10K", "AMAZON_2020_10K", "AMAZON_2019_10K",
    "ADOBE_2021_10K", "EBAY_2022_10K", "ADOBE_2019_10K", "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q", "Pfizer_2023Q2_10Q", "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q", "AMCOR_2022_8K_2022-07-01", "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16", "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20"
]

QUESTION = 'List one material agreement or other exhibit number explicitly identified in the Exhibit Index (e.g., "Exhibit 10.1: Credit Agreement")—or record "none listed" if no material agreements are identified.'
QUESTION_SLUG = "list_one_material_agreement_or_other_exhibit_number_explicit"

# Ground truth
GROUND_TRUTH = {
    "BOEING_2019_10K": "Exhibit 10.1: 364-Day Credit Agreement",
    "ADOBE_2020_10K": "Exhibit 10.1: Credit Agreement",
    "ACTIVISIONBLIZZARD_2020_10K": "Exhibit 10.24: Notice of Stock Option Award",
    "COSTCO_2018_10K": "none listed",
    "AMCOR_2019_10K": "Exhibit 10.1: Transaction Agreement",
    "AMAZON_2020_10K": "Exhibit 10.1: 1997 Stock Incentive Plan",
    "AMAZON_2019_10K": "Exhibit 10.1: 1997 Stock Incentive Plan",
    "ADOBE_2021_10K": "Exhibit 10.1: 2020 Employee Stock Purchase Plan",
    "EBAY_2022_10K": None,
    "ADOBE_2019_10K": "Exhibit 10.1: Credit Agreement",
    "AMCOR_2023Q2_10Q": "None listed",
    "ADOBE_2022Q2_10Q": "Exhibit 10.1: 2019 Equity Incentive Plan, as amended",
    "Pfizer_2023Q2_10Q": "Exhibit 10.1: Executive Officer Severance Policy",
    "ACTIVSIONBLIZZARD_2023Q2_10Q": "Exhibit 10.1: Credit Agreement",
    "3M_2023Q2_10Q": "Exhibit 10.1: Settlement Agreement",
    "AMCOR_2022_8K_2022-07-01": "Exhibit 4.6: Second Supplemental Indenture",
    "COSTCO_2023_8K_dated-2023-08-09": "Exhibit 3.2: Bylaws as amended of Costco Wholesale Corporation",
    "COSTCO_2023_8K_dated-2023-08-16": "Exhibit 99.1: Press release dated August 16, 2023",
    "MGMRESORTS_2023_8K_dated-2023-03-01": "Exhibit 99.1: Press Release of the Company dated March 1, 2023",
    "FOOTLOCKER_2022_8K_dated-2022-05-20": "Exhibit 104: Cover Page Interactive Data File"
}

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def load_doc(doc_name: str):
    path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def rule_exhibit_index(doc: dict) -> list[dict]:
    """Match exhibit index tables and text for 10-K, 10-Q, and 8-K documents."""
    try:
        results = []
        is_8k = "_8K" in doc.get("doc_name", "")

        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")
            page = span.get("page_no", 0)

            if is_8k:
                if "9.01" in path or "financial statements and exhibits" in path:
                    if label == "table":
                        if "exhibit" in text_lower or "description" in text_lower:
                            results.append(span)
                    elif page <= 3:
                        if "exhibit" in text_lower or re.match(r'^\d+\.\d*\.?\s*\w', text):
                            results.append(span)
                        elif label == "section_header" and "9.01" in text_lower:
                            results.append(span)
                elif label == "table" and "9.01" in text_lower and page <= 3:
                    results.append(span)
            else:
                if label == "table":
                    if any(x in path for x in ["item 15", "item 6", "exhibit"]):
                        if "exhibit" in text_lower:
                            results.append(span)

        return results
    except Exception:
        return []


def call_llm_qa(retrieved_text: str, question: str) -> tuple[str, int, int]:
    """Call LLM to answer question based on retrieved text."""
    from models import gpt54

    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=500,
        temperature=0.0,
    )

    answer = (response.choices[0].message.content or "").strip() or "NOT FOUND"
    usage = response.usage
    return answer, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def call_llm_judge(question: str, ground_truth: str, predicted: str) -> tuple[bool, int, int]:
    """Call LLM to judge if predicted answer matches ground truth."""
    from models import gpt54

    judge_system = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the ground truth says "none listed" or similar, accept "none listed", "no material agreements", "Not Found" as correct
- For exhibit answers, focus on the exhibit NUMBER being correct - "Exhibit 10.1" matches if the exhibit number is right
- If the predicted answer is "NOT FOUND" or null, and ground truth has a specific exhibit, judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = predicted if predicted else "null"

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {gt_str}\n"
        f"Predicted: {pred_str}"
    )

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )

    verdict = (response.choices[0].message.content or "").strip().lower()
    usage = response.usage
    correct = verdict == "correct"
    return correct, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def main():
    start_time = time.time()

    print(f"Evaluating rule_exhibit_index on {len(DOCS)} documents\n")
    print(f"Question: {QUESTION}\n")

    results = []
    total_cost = 0
    hits = 0
    valid_docs = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if doc is None:
            print(f"[SKIP] {doc_name} - file not found")
            continue

        valid_docs += 1

        # Apply rule
        spans = rule_exhibit_index(doc)

        # Calculate cost
        full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        # Get ground truth
        answer = GROUND_TRUTH.get(doc_name)

        # Call LLM to get predicted answer
        if spans:
            predicted, qa_in, qa_out = call_llm_qa(retrieved_text, QUESTION)
            total_input_tokens += qa_in
            total_output_tokens += qa_out
            total_llm_calls += 1
        else:
            predicted = "NOT FOUND"
            qa_in, qa_out = 0, 0

        # Judge if correct
        if answer is None:
            # No ground truth - skip judging
            correct = None
            judge_in, judge_out = 0, 0
        else:
            correct, judge_in, judge_out = call_llm_judge(QUESTION, answer, predicted)
            total_input_tokens += judge_in
            total_output_tokens += judge_out
            total_llm_calls += 1

            if correct:
                hits += 1

        status = "HIT" if correct else ("MISS" if correct is False else "N/A")
        print(f"[{status}] {doc_name}: cost={cost:.4f}, spans={len(spans)}")
        if correct is False:
            print(f"    Ground Truth: {answer}")
            print(f"    Predicted:    {predicted}")

        results.append({
            "doc_name": doc_name,
            "ground_truth": answer,
            "predicted": predicted,
            "correct": correct,
            "cost": cost,
            "num_spans": len(spans),
        })

    # Calculate metrics
    docs_with_gt = sum(1 for r in results if r["ground_truth"] is not None)
    accuracy = hits / docs_with_gt if docs_with_gt > 0 else 0
    avg_cost = total_cost / valid_docs if valid_docs > 0 else 0

    latency = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Valid documents: {valid_docs}")
    print(f"  Documents with ground truth: {docs_with_gt}")
    print(f"  Correct predictions: {hits}")
    print(f"  Accuracy (LLM judge): {accuracy:.2%}")
    print(f"  Average cost ratio: {avg_cost:.4f}")
    print(f"  Total latency: {latency:.2f}s")
    print(f"  Total LLM calls: {total_llm_calls}")
    print(f"  Total input tokens: {total_input_tokens}")
    print(f"  Total output tokens: {total_output_tokens}")

    # Save detailed results
    output = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "latency_seconds": round(latency, 2),
        "agent_input_tokens": total_input_tokens,
        "agent_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "num_rules": 1,
        "merge_accuracy": round(accuracy, 4),
        "avg_cost_ratio": round(avg_cost, 6),
        "rules": [
            {
                "rule_name": "rule_exhibit_index",
                "description": "Match exhibit index tables and text for 10-K, 10-Q, and 8-K documents.",
                "coverage": valid_docs,
                "avg_cost_ratio": round(avg_cost, 6),
                "file": f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}/rule_exhibit_index.py"
            }
        ],
        "per_document": results
    }

    output_path = f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}_rule_gen.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
