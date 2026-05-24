#!/usr/bin/env python3
"""Evaluate address rules using LLM judge."""

import json
import os
import sys
import time
import importlib.util

# Add src to path
sys.path.insert(0, "src")

START_TIME = time.time()

# Track token usage
TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_LLM_CALLS = 0

# Sampled documents
SAMPLED_DOCS = [
    "BOEING_2019_10K",
    "ADOBE_2020_10K",
    "ACTIVISIONBLIZZARD_2020_10K",
    "COSTCO_2018_10K",
    "AMCOR_2019_10K",
    "AMAZON_2020_10K",
    "AMAZON_2019_10K",
    "ADOBE_2021_10K",
    "EBAY_2022_10K",
    "ADOBE_2019_10K",
    "AMCOR_2023Q2_10Q",
    "ADOBE_2022Q2_10Q",
    "Pfizer_2023Q2_10Q",
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "MGMRESORTS_2023_8K_dated-2023-03-01",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is the address of principal executive offices and ZIP code?"
QUESTION_SLUG = "what_is_the_address_of_principal_executive_offices_and_zip_c"

# Load labels
with open("data/financebench/sample_mix_doc_labels.json") as f:
    labels = json.load(f)

# Build ground truth dict
ground_truth = {}
for doc_name in SAMPLED_DOCS:
    pdf_name = doc_name + ".pdf"
    if pdf_name in labels and QUESTION in labels[pdf_name]:
        ground_truth[doc_name] = labels[pdf_name][QUESTION]

# Load rules
RULES_DIR = f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}"
rules = []

for fname in os.listdir(RULES_DIR):
    if fname.endswith(".py") and fname.startswith("rule_"):
        fpath = os.path.join(RULES_DIR, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            if name.startswith("rule_"):
                rules.append((name, getattr(module, name), fpath))

print(f"Loaded {len(rules)} rules: {[r[0] for r in rules]}")

# Load Azure model for QA and judge
try:
    from models import azure
    model = azure
    print("Using Azure model for LLM calls")
except ImportError:
    print("Azure model not available, trying gpt54...")
    from models import gpt54
    model = gpt54

def call_qa_llm(context: str, question: str) -> tuple[str, int, int]:
    """Call LLM to answer question based on context."""
    global TOTAL_LLM_CALLS, TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS

    try:
        response = model.gpt_54(question, context)
        TOTAL_LLM_CALLS += 1
        input_tokens = len(context.split()) + len(question.split())
        output_tokens = len(response.split())
        TOTAL_INPUT_TOKENS += int(input_tokens * 1.3)
        TOTAL_OUTPUT_TOKENS += int(output_tokens * 1.3)
        return response, int(input_tokens * 1.3), int(output_tokens * 1.3)
    except Exception as e:
        print(f"Error calling QA LLM: {e}")
        return "NOT FOUND", 0, 0

def call_judge_llm(question: str, predicted: str, ground_truth: str) -> tuple[bool, int, int]:
    """Call LLM judge to check if answer is correct."""
    global TOTAL_LLM_CALLS, TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS

    prompt = f"""You are an answer equivalence judge for a financial document QA system.

Question: {question}

Predicted Answer: {predicted}

Ground Truth Answer: {ground_truth}

Judge whether the predicted answer is CORRECT (semantically equivalent to ground truth, ignoring minor formatting differences) or INCORRECT.

For address questions, consider:
- Street addresses with same numbers and street name are equivalent
- State abbreviations (CA vs California) are equivalent
- Minor punctuation differences are acceptable
- ZIP codes should match

Reply with exactly one word: CORRECT or INCORRECT"""

    try:
        response = model.chat_completions(prompt)
        TOTAL_LLM_CALLS += 1
        input_tokens = len(prompt.split())
        output_tokens = len(response.split())
        TOTAL_INPUT_TOKENS += int(input_tokens * 1.3)
        TOTAL_OUTPUT_TOKENS += int(output_tokens * 1.3)
        return "CORRECT" in response.upper(), int(input_tokens * 1.3), int(output_tokens * 1.3)
    except Exception as e:
        print(f"Error calling judge LLM: {e}")
        return False, 0, 0

# Evaluate each document
correct_count = 0
total_count = 0
cost_sum = 0
cost_count = 0
results = []
rule_coverage = {r[0]: 0 for r in rules}
rule_costs = {r[0]: [] for r in rules}

for doc_name in SAMPLED_DOCS:
    doc_path = f"data/financebench/processing/{doc_name}_reconstructed.json"
    if not os.path.exists(doc_path):
        print(f"SKIP (missing): {doc_name}")
        continue

    with open(doc_path) as f:
        doc = json.load(f)

    answer = ground_truth.get(doc_name, "")
    if not answer:
        print(f"SKIP (no answer): {doc_name}")
        continue

    total_count += 1

    # Apply all rules and merge
    retrieved_spans = []
    rules_hit = []
    for rule_name, rule_func, rule_path in rules:
        spans = rule_func(doc)
        if spans:
            rules_hit.append(rule_name)
            rule_coverage[rule_name] += 1
            retrieved_spans.extend(spans)
            # Track per-rule cost
            span_text = " ".join(s.get("text", "") for s in spans)
            full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
            if len(full_text) > 0:
                rule_costs[rule_name].append(len(span_text) / len(full_text))

    # Dedupe by text
    seen = set()
    unique_spans = []
    for s in retrieved_spans:
        text = s.get("text", "")
        if text not in seen:
            seen.add(text)
            unique_spans.append(s)

    # Concatenate retrieved text
    retrieved_text = " ".join(s.get("text", "") for s in unique_spans)

    # Full doc text
    full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))

    # Cost ratio
    if len(full_text) > 0:
        cost = len(retrieved_text) / len(full_text)
        cost_sum += cost
        cost_count += 1
    else:
        cost = 0

    # Call LLM to get answer
    predicted_answer, qa_in, qa_out = call_qa_llm(retrieved_text, QUESTION)

    # Call judge
    is_correct, judge_in, judge_out = call_judge_llm(QUESTION, predicted_answer, answer)

    if is_correct:
        correct_count += 1
        status = "CORRECT"
    else:
        status = "WRONG"

    results.append({
        "doc_name": doc_name,
        "status": status,
        "ground_truth": answer,
        "predicted": predicted_answer,
        "cost": cost,
        "rules_hit": rules_hit,
    })

    print(f"{status}: {doc_name} (cost={cost:.4f})")
    if status == "WRONG":
        print(f"  GT: {answer}")
        print(f"  PRED: {predicted_answer[:200]}...")

merge_accuracy = correct_count / total_count if total_count > 0 else 0
avg_cost = cost_sum / cost_count if cost_count > 0 else 0

print(f"\n{'='*60}")
print(f"Total: {total_count}")
print(f"Correct: {correct_count}")
print(f"Merge Accuracy: {merge_accuracy:.2%}")
print(f"Avg Cost: {avg_cost:.4f}")
print(f"Total LLM calls: {TOTAL_LLM_CALLS}")
print(f"Total Input Tokens: {TOTAL_INPUT_TOKENS}")
print(f"Total Output Tokens: {TOTAL_OUTPUT_TOKENS}")
print(f"Latency: {time.time() - START_TIME:.2f}s")

print(f"\n{'='*60}")
print("Per-Rule Stats:")
for rule_name, _, rule_path in rules:
    coverage = rule_coverage[rule_name]
    costs = rule_costs[rule_name]
    avg_rule_cost = sum(costs) / len(costs) if costs else 0
    print(f"  {rule_name}: coverage={coverage}, avg_cost={avg_rule_cost:.4f}")

# Save results
output = {
    "question": QUESTION,
    "question_slug": QUESTION_SLUG,
    "merge_accuracy": merge_accuracy,
    "avg_cost_ratio": avg_cost,
    "total_docs": total_count,
    "correct_docs": correct_count,
    "results": results,
}

with open(f"rules/agent/financebench_mix_doc_claude/{QUESTION_SLUG}/eval_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved results to {QUESTION_SLUG}/eval_results.json")
