#!/usr/bin/env python3
"""Evaluate shares outstanding rules with LLM judge."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for rule loading
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tiktoken
from openai import AzureOpenAI
from azure_local import load_azure_credentials_from_local

# Load Azure credentials
_AZURE_JSON = _ROOT / "local" / "azure.json"
api_key, AZURE_API_VERSION, AZURE_ENDPOINT, _deployment = load_azure_credentials_from_local(_AZURE_JSON)
AZURE_DEPLOYMENT = (_deployment or "gpt-4o").strip()

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=api_key,
)

# Configuration
QUESTION = "How many shares of common stock were outstanding as of the cover-page reference date?"
QUESTION_SLUG = "how_many_shares_of_common_stock_were_outstanding_as_of_the_c"

DOC_NAMES = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
]

RULES_DIR = Path("rules/agent/financebench_agent") / QUESTION_SLUG
LABELS_FILE = Path("data/financebench/sample_doc_labels.json")
PROCESSING_DIR = Path("data/financebench/processing")

# Tracking variables
start_time = time.time()
total_input_tokens = 0
total_output_tokens = 0
total_llm_calls = 0

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# Initialize Anthropic client
client = anthropic.Anthropic()

def llm_answer(retrieved_text: str, question: str) -> tuple[str, int, int]:
    """Ask LLM to answer the question from retrieved text."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )
    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=100,
        temperature=0.0,
    )

    total_llm_calls += 1
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    answer = response.choices[0].message.content.strip()
    return answer, input_tokens, output_tokens

def llm_judge(question: str, predicted: str, ground_truth: str) -> tuple[bool, int, int]:
    """Use LLM to judge if predicted answer matches ground truth."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    judge_system = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- Numbers like "1,625,907,855" and "1625907855" are equivalent
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=10,
        temperature=0.0,
    )

    total_llm_calls += 1
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    verdict = response.choices[0].message.content.strip().upper()
    return verdict == "CORRECT", input_tokens, output_tokens

def load_rule_fn(rule_file: Path):
    """Load a rule function from a file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if name.startswith("rule_") and callable(obj):
            return obj
    raise ValueError(f"No rule_* function found in {rule_file}")

def main():
    global start_time

    # Load ground truth
    labels = json.loads(LABELS_FILE.read_text())
    ground_truth = {}
    for doc_name in DOC_NAMES:
        doc_key = f"{doc_name}.pdf"
        if doc_key in labels:
            ground_truth[doc_name] = labels[doc_key][QUESTION]

    # Load rules
    rule_files = sorted(RULES_DIR.glob("*.py"))
    rules = []
    for rf in rule_files:
        rules.append((rf.stem, load_rule_fn(rf)))

    print(f"Loaded {len(rules)} rules:")
    for name, _ in rules:
        print(f"  - {name}")

    # Evaluate each document
    results = []
    correct_count = 0
    total_cost = 0.0

    rule_coverage = {name: 0 for name, _ in rules}
    rule_costs = {name: [] for name, _ in rules}

    print("\n" + "="*70)
    print("Evaluating with LLM judge")
    print("="*70 + "\n")

    for doc_name in DOC_NAMES:
        doc_path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        doc = json.loads(doc_path.read_text())
        gt = ground_truth[doc_name]

        # Apply all rules and union spans
        all_spans = []
        for rule_name, rule_fn in rules:
            spans = rule_fn(doc)
            if spans:
                rule_coverage[rule_name] += 1
                all_spans.extend(spans)

        # Deduplicate by text
        seen = set()
        unique_spans = []
        for s in all_spans:
            key = s.get("text", "")
            if key not in seen:
                seen.add(key)
                unique_spans.append(s)

        # Compute cost
        retrieved_text = " ".join(s.get("text", "") for s in unique_spans)
        doc_text = " ".join(s.get("text", "") for s in doc.get("texts", []))

        retrieved_tokens = count_tokens(retrieved_text)
        doc_tokens = count_tokens(doc_text)
        cost_ratio = retrieved_tokens / doc_tokens if doc_tokens > 0 else 0
        total_cost += cost_ratio

        # Track per-rule costs
        for rule_name, rule_fn in rules:
            spans = rule_fn(doc)
            if spans:
                rule_text = " ".join(s.get("text", "") for s in spans)
                rule_tokens = count_tokens(rule_text)
                rule_costs[rule_name].append(rule_tokens / doc_tokens if doc_tokens > 0 else 0)

        # Get LLM answer
        predicted, qa_in, qa_out = llm_answer(retrieved_text, QUESTION)

        # Judge answer
        correct, judge_in, judge_out = llm_judge(QUESTION, predicted, gt)

        if correct:
            correct_count += 1

        status = "CORRECT" if correct else "WRONG"
        print(f"{doc_name}: {status}")
        print(f"  Ground truth: {gt}")
        print(f"  Predicted: {predicted}")
        print(f"  Cost ratio: {cost_ratio:.6f}")
        print()

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": correct,
            "cost_ratio": cost_ratio,
            "retrieved_tokens": retrieved_tokens,
            "doc_tokens": doc_tokens,
        })

    # Summary statistics
    merge_accuracy = correct_count / len(DOC_NAMES)
    avg_cost = total_cost / len(DOC_NAMES)
    latency = time.time() - start_time

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Merge accuracy: {merge_accuracy:.2%} ({correct_count}/{len(DOC_NAMES)})")
    print(f"Average cost ratio: {avg_cost:.6f}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Latency: {latency:.2f}s")
    print()
    print("Per-rule statistics:")
    for rule_name, _ in rules:
        coverage = rule_coverage[rule_name]
        costs = rule_costs[rule_name]
        avg_rule_cost = sum(costs) / len(costs) if costs else 0
        print(f"  {rule_name}: coverage={coverage}/{len(DOC_NAMES)}, avg_cost={avg_rule_cost:.6f}")

    # Generate log file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_data = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(latency, 2),
        "agent_input_tokens": total_input_tokens,
        "agent_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "num_rules": len(rules),
        "merge_accuracy": round(merge_accuracy, 4),
        "avg_cost_ratio": round(avg_cost, 6),
        "rules": [
            {
                "rule_name": rule_name,
                "description": rule_fn.__doc__ or "",
                "coverage": rule_coverage[rule_name],
                "avg_cost_ratio": round(sum(rule_costs[rule_name]) / len(rule_costs[rule_name]), 6) if rule_costs[rule_name] else 0,
                "file": str(RULES_DIR / f"{rule_name}.py"),
            }
            for rule_name, rule_fn in rules
        ],
        "per_document": results,
    }

    log_file = RULES_DIR.parent / f"{QUESTION_SLUG}_claude-opus-4-5_{timestamp}_rule_gen.json"
    log_file.write_text(json.dumps(log_data, indent=2, ensure_ascii=False))
    print(f"\nLog saved to: {log_file}")

    return merge_accuracy, avg_cost

if __name__ == "__main__":
    accuracy, cost = main()
    print(f"\nFinal: accuracy={accuracy:.2%}, avg_cost={cost:.6f}")
