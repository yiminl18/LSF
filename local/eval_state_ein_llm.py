#!/usr/bin/env python3
"""LLM-judge evaluation for state/EIN rules."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

START_TIME = time.time()
INPUT_TOKENS = 0
OUTPUT_TOKENS = 0
LLM_CALLS = 0

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
QUESTION_SLUG = "what_is_the_state__or_other_jurisdiction__of_incorporation_a"


def rule_page1_state_ein(doc: dict) -> list[dict]:
    """Match page 1 spans containing state of incorporation and EIN information."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        ein_pattern = re.compile(r'\d{2}-\d{7}')

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

            if "state or other jurisdiction" in text_lower:
                results.append(span)
                for j in range(1, 4):
                    if i - j >= 0 and texts[i-j].get("page_no") == 1:
                        prev_text = texts[i-j].get("text", "").lower().strip()
                        if prev_text in jurisdictions:
                            results.append(texts[i-j])
                            break
                        for state in jurisdictions:
                            if state in prev_text and len(prev_text) < 50:
                                results.append(texts[i-j])
                                break

            if "employer identification" in text_lower:
                results.append(span)
                if not ein_pattern.search(text) and i > 0 and texts[i-1].get("page_no") == 1:
                    results.append(texts[i-1])

            if ein_pattern.search(text):
                results.append(span)

            if text_lower in jurisdictions:
                results.append(span)

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


def get_doc_text(doc):
    return " ".join(s.get("text", "") for s in doc.get("texts", []))


def get_span_text(spans):
    return " ".join(s.get("text", "") for s in spans)


def tiktoken_count(text):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def llm_answer(context, question):
    """Call LLM to answer the question based on context."""
    global INPUT_TOKENS, OUTPUT_TOKENS, LLM_CALLS
    try:
        from models import gpt54

        system = "You are a financial document analyst. Answer the question based solely on the provided context. Be concise and precise."
        prompt = f"""Context from SEC filing:
{context}

Question: {question}

Answer:"""

        response = gpt54.chat_completions(prompt, system=system, max_completion_tokens=200)
        LLM_CALLS += 1
        INPUT_TOKENS += tiktoken_count(system + prompt)
        OUTPUT_TOKENS += tiktoken_count(response)
        return response.strip()
    except Exception as e:
        print(f"LLM answer error: {e}")
        return "NOT FOUND"


def llm_judge(question, predicted, ground_truth):
    """Call LLM to judge if predicted answer matches ground truth."""
    global INPUT_TOKENS, OUTPUT_TOKENS, LLM_CALLS
    try:
        from models import gpt54

        system = """You are an answer equivalence judge for a financial document QA system.
Judge whether the predicted answer is semantically equivalent to the ground truth.
- Treat "Delaware" and "State of Delaware" as equivalent
- Treat "13-3513936" and "EIN 13-3513936" as equivalent
- Ignore minor formatting differences
Reply with exactly one word: CORRECT or INCORRECT"""

        prompt = f"""Question: {question}
Predicted Answer: {predicted}
Ground Truth: {ground_truth}

Judgment:"""

        response = gpt54.chat_completions(prompt, system=system, max_completion_tokens=10)
        LLM_CALLS += 1
        INPUT_TOKENS += tiktoken_count(system + prompt)
        OUTPUT_TOKENS += tiktoken_count(response)
        return "CORRECT" in response.upper()
    except Exception as e:
        print(f"LLM judge error: {e}")
        # Fallback to substring match
        gt_parts = re.split(r'[,;]\s*', ground_truth.lower())
        pred_lower = predicted.lower()
        return all(p.strip() in pred_lower for p in gt_parts if p.strip())


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

# Evaluate
correct = 0
total_cost = 0
results = []

for doc_name in docs:
    if doc_name not in ground_truth:
        continue

    doc = docs[doc_name]
    gt = ground_truth[doc_name]

    # Apply rule
    spans = rule_page1_state_ein(doc)
    retrieved_text = get_span_text(spans)
    full_text = get_doc_text(doc)

    # Calculate cost
    if full_text:
        cost = tiktoken_count(retrieved_text) / tiktoken_count(full_text)
    else:
        cost = 0
    total_cost += cost

    # LLM answer
    predicted = llm_answer(retrieved_text, QUESTION)
    print(f"\n{doc_name}:")
    print(f"  GT: {gt}")
    print(f"  Predicted: {predicted}")

    # LLM judge
    is_correct = llm_judge(QUESTION, predicted, gt)
    print(f"  Correct: {is_correct}")

    if is_correct:
        correct += 1

    results.append({
        "doc_name": doc_name,
        "ground_truth": gt,
        "predicted": predicted,
        "correct": is_correct,
        "cost_ratio": cost,
    })

n_docs = len([d for d in docs if d in ground_truth])
merge_accuracy = correct / n_docs if n_docs > 0 else 0
avg_cost = total_cost / n_docs if n_docs > 0 else 0

print(f"\n=== FINAL RESULTS ===")
print(f"Merge Accuracy: {correct}/{n_docs} = {merge_accuracy:.2%}")
print(f"Avg Cost Ratio: {avg_cost:.4f}")
print(f"Total LLM Calls: {LLM_CALLS}")
print(f"Total Input Tokens: {INPUT_TOKENS}")
print(f"Total Output Tokens: {OUTPUT_TOKENS}")

# Save results
latency = time.time() - START_TIME
timestamp = datetime.now(timezone.utc).isoformat()

output = {
    "question": QUESTION,
    "question_slug": QUESTION_SLUG,
    "timestamp": timestamp,
    "latency_seconds": latency,
    "agent_input_tokens": INPUT_TOKENS,
    "agent_output_tokens": OUTPUT_TOKENS,
    "total_llm_calls": LLM_CALLS,
    "num_rules": 1,
    "merge_accuracy": merge_accuracy,
    "avg_cost_ratio": avg_cost,
    "rules": [
        {
            "rule_name": "rule_page1_state_ein",
            "description": "Match page 1 spans containing state of incorporation and EIN information.",
            "coverage": n_docs,
            "avg_cost_ratio": avg_cost,
            "file": f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}/rule_page1_state_ein.py"
        }
    ],
    "per_document_results": results
}

output_file = f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}/{QUESTION_SLUG}_claude-opus-4-5_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rule_gen.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_file}")
print(f"Total time: {latency:.2f}s")
