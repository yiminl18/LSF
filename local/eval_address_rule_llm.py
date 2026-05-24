#!/usr/bin/env python3
"""Full LLM-judge evaluation for address rule."""

import json
import sys
import time
from pathlib import Path

# Add src to path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tiktoken
from models import gpt54

# Configuration
docs = ['AMCOR_2019_10K', 'COSTCO_2017_10K', 'BOEING_2018_10K', 'AMAZON_2018_10K', 'EBAY_2021_10K',
        'AMAZON_2016_10K', 'CORNING_2022_10K', 'NIKE_2021_10K', 'LOCKHEEDMARTIN_2022_10K', 'JOHNSON_JOHNSON_2022_10K']

question = 'What is the address of principal executive offices and ZIP code?'
question_slug = 'what_is_the_address_of_principal_executive_offices_and_zip_c'

# Load labels
with open('data/financebench/sample_doc_labels.json') as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in docs:
    ground_truth[doc_name] = labels[f'{doc_name}.pdf'][question]


def rule_page1_address_principal(doc: dict) -> list[dict]:
    """Match page 1 spans containing or near 'address of principal executive offices'."""
    try:
        results = []
        texts = doc.get("texts", [])
        seen_ids = set()

        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue

            text = span.get("text", "").lower()
            text_span = span.get("text_span", "").lower()

            if "address of principal" in text_span or "address and telephone" in text_span:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))

            if "address of principal" in text or "address and telephone" in text:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                for offset in [1, 2, 3]:
                    prev_idx = i - offset
                    if prev_idx >= 0 and texts[prev_idx].get("page_no") == 1:
                        prev = texts[prev_idx]
                        if id(prev) not in seen_ids:
                            prev_text = prev.get("text", "").lower()
                            if "jurisdiction" in prev_text or "i.r.s." in prev_text or "employer identification" in prev_text:
                                continue
                            results.append(prev)
                            seen_ids.add(id(prev))

            if "(zip code)" in text:
                if id(span) not in seen_ids:
                    results.append(span)
                    seen_ids.add(id(span))
                if i > 0 and texts[i-1].get("page_no") == 1:
                    prev = texts[i-1]
                    if id(prev) not in seen_ids:
                        results.append(prev)
                        seen_ids.add(id(prev))

        return results
    except Exception:
        return []


# Judge system prompt
JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- Treat different formatting of addresses as equivalent (commas vs newlines)
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

# QA system prompt
QA_SYSTEM = """\
You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply with "NOT FOUND".
Return only the answer — the complete address with ZIP code, not a full sentence."""

enc = tiktoken.encoding_for_model("gpt-4")

# Token tracking
total_qa_input_tokens = 0
total_qa_output_tokens = 0
total_judge_input_tokens = 0
total_judge_output_tokens = 0
total_llm_calls = 0

start_time = time.time()

results = []
correct_count = 0
total_cost = 0

print("Evaluating rule_page1_address_principal with LLM judge\n")

for doc_name in docs:
    print(f"Processing {doc_name}...")

    with open(f'data/financebench/processing/{doc_name}_reconstructed.json') as f:
        doc = json.load(f)

    # Apply rule
    retrieved = rule_page1_address_principal(doc)

    # Get retrieved text
    retrieved_text = "\n\n".join([s.get("text", "") for s in retrieved])

    # Get full document text
    full_text = " ".join([s.get("text", "") for s in doc["texts"]])

    # Calculate cost
    retrieved_tokens = len(enc.encode(retrieved_text))
    full_tokens = len(enc.encode(full_text))
    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost

    gt = ground_truth[doc_name]

    # Step 1: QA call
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    t0 = time.time()
    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=200,
        temperature=0.0,
    )
    qa_latency = time.time() - t0

    predicted_answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    total_qa_input_tokens += usage.prompt_tokens if usage else 0
    total_qa_output_tokens += usage.completion_tokens if usage else 0
    total_llm_calls += 1

    # Step 2: Judge call
    judge_prompt = f"Question: {question}\nGround Truth: {gt}\nPredicted: {predicted_answer}"

    t0 = time.time()
    judge_response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_prompt},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    judge_latency = time.time() - t0

    judge_verdict = (judge_response.choices[0].message.content or "").strip().lower()
    usage = judge_response.usage
    total_judge_input_tokens += usage.prompt_tokens if usage else 0
    total_judge_output_tokens += usage.completion_tokens if usage else 0
    total_llm_calls += 1

    is_correct = judge_verdict == "correct"
    if is_correct:
        correct_count += 1

    results.append({
        "doc_name": doc_name,
        "ground_truth": gt,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "judge_verdict": judge_verdict,
        "cost_ratio": cost,
        "retrieved_tokens": retrieved_tokens,
        "full_tokens": full_tokens,
        "num_spans": len(retrieved),
    })

    status = "CORRECT" if is_correct else "INCORRECT"
    print(f"  {status}: GT='{gt[:50]}...' PRED='{predicted_answer[:50]}...'")

total_time = time.time() - start_time

# Summary
merge_accuracy = correct_count / len(docs)
avg_cost = total_cost / len(docs)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Merge accuracy: {correct_count}/{len(docs)} = {merge_accuracy:.2%}")
print(f"Average cost ratio: {avg_cost:.4f}")
print(f"Total latency: {total_time:.2f}s")
print(f"Total LLM calls: {total_llm_calls}")
print(f"QA tokens: {total_qa_input_tokens} in / {total_qa_output_tokens} out")
print(f"Judge tokens: {total_judge_input_tokens} in / {total_judge_output_tokens} out")

# Print failures
failures = [r for r in results if not r["is_correct"]]
if failures:
    print(f"\nFailed documents ({len(failures)}):")
    for f in failures:
        print(f"  {f['doc_name']}")
        print(f"    GT: {f['ground_truth']}")
        print(f"    Pred: {f['predicted_answer']}")
        print(f"    Judge: {f['judge_verdict']}")

# Save results
output = {
    "question": question,
    "question_slug": question_slug,
    "rule_name": "rule_page1_address_principal",
    "merge_accuracy": merge_accuracy,
    "avg_cost_ratio": avg_cost,
    "total_latency_seconds": total_time,
    "total_llm_calls": total_llm_calls,
    "qa_input_tokens": total_qa_input_tokens,
    "qa_output_tokens": total_qa_output_tokens,
    "judge_input_tokens": total_judge_input_tokens,
    "judge_output_tokens": total_judge_output_tokens,
    "results": results,
}

with open("local/eval_address_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to local/eval_address_results.json")
