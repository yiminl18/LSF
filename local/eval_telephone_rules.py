#!/usr/bin/env python3
"""Evaluate telephone rules with LLM judge."""
import json
import re
import sys
import time
from pathlib import Path

# Add src to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tiktoken
import importlib.util

def load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))


# Load model client directly
from azure_local import load_azure_credentials_from_local
from openai import AzureOpenAI

_AZURE_JSON = _ROOT / "local" / "azure.json"
api_key, AZURE_API_VERSION, AZURE_ENDPOINT, _deployment = load_azure_credentials_from_local(_AZURE_JSON)
AZURE_DEPLOYMENT = (_deployment or "gpt-5.4").strip()

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=api_key,
)

JUDGE_SYSTEM = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat phone numbers with different formatting as equivalent if digits match
- "(408) 536-6000" and "408-536-6000" and "4085366000" are all equivalent
- "+44 117 9753200" and "+441179753200" are equivalent
- Ignore leading/trailing whitespace, punctuation, parentheses, and dashes
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

def call_llm(system: str, user: str, max_tokens: int = 100) -> tuple[str, dict]:
    """Call LLM and return response and usage dict."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    r = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=0.0
    )
    usage = {
        "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
        "completion_tokens": r.usage.completion_tokens if r.usage else 0
    }
    return (r.choices[0].message.content or "").strip(), usage


def judge_answer(question: str, predicted: str, ground_truth: str) -> tuple[bool, dict]:
    """Use LLM to judge if predicted matches ground truth."""
    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )
    
    response, usage = call_llm(JUDGE_SYSTEM, user_prompt, max_tokens=10)
    return response.strip().upper() == "CORRECT", usage


def qa_with_context(question: str, context: str) -> tuple[str, dict]:
    """Ask the LLM to answer based on the retrieved context."""
    system = "Answer the question based only on the provided context. Be concise."
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return call_llm(system, user, max_tokens=100)


# All sampled documents
sampled_docs = [
    'BOEING_2019_10K',
    'ADOBE_2020_10K',
    'ACTIVISIONBLIZZARD_2020_10K',
    'COSTCO_2018_10K',
    'AMCOR_2019_10K',
    'AMAZON_2020_10K',
    'AMAZON_2019_10K',
    'ADOBE_2021_10K',
    'EBAY_2022_10K',
    'ADOBE_2019_10K',
    'AMCOR_2023Q2_10Q',
    'ADOBE_2022Q2_10Q',
    'ACTIVSIONBLIZZARD_2023Q2_10Q',
    '3M_2023Q2_10Q',
    'AMCOR_2022_8K_2022-07-01',
    'COSTCO_2023_8K_dated-2023-08-09',
    'COSTCO_2023_8K_dated-2023-08-16',
    'FOOTLOCKER_2022_8K_dated-2022-05-20'
]

question = "What is the registrant's telephone number?"

# Load ground truth
with open(_ROOT / 'data/financebench/sample_mix_doc_labels.json') as f:
    labels = json.load(f)

ground_truth = {}
for doc_name in sampled_docs:
    pdf_name = doc_name + '.pdf'
    if pdf_name in labels:
        answer = labels[pdf_name].get(question)
        if answer:
            ground_truth[doc_name] = answer

# Load rule
rule_file = _ROOT / 'rules/agent/financebench_mix_doc_claude/what_is_the_registrant_s_telephone_number/rule_page1_telephone_or_phone.py'
rule_fn = load_rule_fn(rule_file)

enc = tiktoken.get_encoding("cl100k_base")

# Run evaluation
start_time = time.time()
total_input_tokens = 0
total_output_tokens = 0
total_llm_calls = 0

correct_count = 0
total_cost = 0
results = []

print(f"Evaluating {len(sampled_docs)} documents...\n")

for doc_name in sampled_docs:
    path = _ROOT / f'data/financebench/processing/{doc_name}_reconstructed.json'
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception as e:
        print(f"Error loading {doc_name}: {e}")
        continue

    gt = ground_truth.get(doc_name, '')
    
    # Apply rule
    retrieved = rule_fn(doc)
    
    # Get retrieved text
    retrieved_text = "\n".join(s.get("text", "") for s in retrieved)
    
    # Calculate cost
    full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))
    retrieved_tokens = len(enc.encode(retrieved_text))
    full_tokens = len(enc.encode(full_text))
    cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
    total_cost += cost
    
    # QA with LLM
    predicted, qa_usage = qa_with_context(question, retrieved_text)
    total_input_tokens += qa_usage.get("prompt_tokens", 0)
    total_output_tokens += qa_usage.get("completion_tokens", 0)
    total_llm_calls += 1
    
    # Judge with LLM
    is_correct, judge_usage = judge_answer(question, predicted, gt)
    total_input_tokens += judge_usage.get("prompt_tokens", 0)
    total_output_tokens += judge_usage.get("completion_tokens", 0)
    total_llm_calls += 1
    
    if is_correct:
        correct_count += 1
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {doc_name}")
    print(f"   GT: {gt}")
    print(f"   Predicted: {predicted}")
    print(f"   Cost: {cost:.4f}")
    print()
    
    results.append({
        "doc_name": doc_name,
        "ground_truth": gt,
        "predicted": predicted,
        "is_correct": is_correct,
        "cost": cost,
        "retrieved_spans": len(retrieved)
    })

end_time = time.time()

merge_accuracy = correct_count / len(sampled_docs)
avg_cost = total_cost / len(sampled_docs)

print("\n" + "="*60)
print(f"RESULTS")
print(f"="*60)
print(f"Merge Accuracy: {correct_count}/{len(sampled_docs)} = {merge_accuracy:.2%}")
print(f"Avg Cost Ratio: {avg_cost:.4f}")
print(f"Latency: {end_time - start_time:.2f} seconds")
print(f"Total LLM Calls: {total_llm_calls}")
print(f"Total Input Tokens: {total_input_tokens}")
print(f"Total Output Tokens: {total_output_tokens}")
print(f"="*60)

# Save results
output = {
    "question": question,
    "merge_accuracy": merge_accuracy,
    "avg_cost_ratio": avg_cost,
    "latency_seconds": end_time - start_time,
    "total_llm_calls": total_llm_calls,
    "total_input_tokens": total_input_tokens,
    "total_output_tokens": total_output_tokens,
    "results": results
}

with open(_ROOT / 'local/eval_telephone_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to local/eval_telephone_results.json")
