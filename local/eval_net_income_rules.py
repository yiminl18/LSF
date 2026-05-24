#!/usr/bin/env python3
"""Evaluate net income rules with LLM-as-a-judge using Anthropic API."""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("anthropic package not installed")
    sys.exit(1)

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text):
        return int(len(text.split()) * 1.3)

# Constants
SAMPLED_DOCS = [
    "AMCOR_2019_10K",
    "COSTCO_2017_10K",
    "BOEING_2018_10K",
    "AMAZON_2018_10K",
    "EBAY_2021_10K",
    "AMAZON_2016_10K",
    "CORNING_2022_10K",
    "NIKE_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "JOHNSON_JOHNSON_2022_10K"
]

QUESTION = "What is net income (loss) for the most recent fiscal year?"
QUESTION_SLUG = "what_is_net_income__loss__for_the_most_recent_fiscal_year"
PROCESSING_DIR = Path("/home/yiminglin/LSF/data/financebench/processing")
LABELS_FILE = Path("/home/yiminglin/LSF/data/financebench/sample_doc_labels.json")
RULES_DIR = Path("/home/yiminglin/LSF/rules/agent/financebench_agent") / QUESTION_SLUG

# Global token tracking
total_input_tokens = 0
total_output_tokens = 0
total_llm_calls = 0

client = anthropic.Anthropic()

def load_rule_function(rule_path):
    """Dynamically load a rule function from a .py file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rule_module", rule_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in vars(mod).items():
        if name.startswith("rule_") and callable(obj):
            return obj
    raise ValueError(f"No rule function found in {rule_path}")

def load_documents():
    docs = {}
    for doc_name in SAMPLED_DOCS:
        path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
        if path.exists():
            with open(path) as f:
                docs[doc_name] = json.load(f)
    return docs

def load_ground_truth():
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        key = f"{doc_name}.pdf"
        if key in labels and QUESTION in labels[key]:
            ground_truth[doc_name] = labels[key][QUESTION]
    return ground_truth

def call_llm_qa(retrieved_text, question):
    """Call Claude to answer the question based on retrieved text."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = """\
You are a financial document QA assistant.
You are given a passage extracted from a financial filing and a question.
Answer the question using only the provided passage.
If the passage does not contain enough information to answer, reply with "NOT FOUND".
Return only the answer — a short value or phrase, not a full sentence."""

    user_prompt = f"Passage:\n{retrieved_text}\n\nQuestion: {question}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens
    total_llm_calls += 1

    return response.content[0].text.strip()

def call_llm_judge(question, predicted, ground_truth):
    """Call Claude to judge if the predicted answer is correct."""
    global total_input_tokens, total_output_tokens, total_llm_calls

    system_prompt = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect

Reply with exactly one word: CORRECT or INCORRECT"""

    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    user_prompt = f"Question: {question}\nGround Truth: {gt_str}\nPredicted: {predicted}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens
    total_llm_calls += 1

    verdict = response.content[0].text.strip().lower()
    return verdict == "correct"

def compute_cost(retrieved_spans, doc):
    full_text = " ".join(span.get("text", "") for span in doc.get("texts", []))
    retrieved_text = " ".join(span.get("text", "") for span in retrieved_spans)
    full_tokens = count_tokens(full_text)
    retrieved_tokens = count_tokens(retrieved_text)
    if full_tokens == 0:
        return 0.0
    return retrieved_tokens / full_tokens

def main():
    global total_input_tokens, total_output_tokens, total_llm_calls

    session_start = time.time()

    # Load data
    docs = load_documents()
    ground_truth = load_ground_truth()

    # Find all rule files
    rule_files = sorted(RULES_DIR.glob("*.py"))
    if not rule_files:
        print(f"No rule files found in {RULES_DIR}")
        return

    print(f"Found {len(rule_files)} rule(s) in {RULES_DIR}")

    # Load all rules
    rules = []
    for rule_file in rule_files:
        rule_fn = load_rule_function(rule_file)
        rules.append({
            "name": rule_file.stem,
            "file": str(rule_file),
            "fn": rule_fn,
            "hits": 0,
            "costs": []
        })

    print("\n" + "=" * 80)
    print("EVALUATING WITH LLM JUDGE")
    print("=" * 80)

    per_document = []
    for doc_name in SAMPLED_DOCS:
        if doc_name not in docs:
            continue
        doc = docs[doc_name]
        answer = ground_truth.get(doc_name, "")

        # Apply all rules and union spans
        all_spans = []
        seen_indices = set()
        for rule in rules:
            spans = rule["fn"](doc)
            for span in spans:
                idx = id(span)
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    all_spans.append(span)
            if spans:
                rule["hits"] += 1
                rule["costs"].append(compute_cost(spans, doc))

        # Sort by page order
        sorted_spans = sorted(all_spans, key=lambda s: (s.get("page_no", 0), 0))
        retrieved_text = "\n\n".join(s.get("text", "") for s in sorted_spans)

        cost = compute_cost(sorted_spans, doc)

        # Call LLM to answer
        if retrieved_text.strip():
            predicted = call_llm_qa(retrieved_text, QUESTION)
        else:
            predicted = "NOT FOUND"

        # Call LLM judge
        correct = call_llm_judge(QUESTION, predicted, answer)

        print(f"  {doc_name}: correct={correct}, cost={cost:.4f}")
        print(f"    Answer: {answer}")
        print(f"    Predicted: {predicted[:80]}...")

        per_document.append({
            "doc_name": doc_name,
            "ground_truth": answer,
            "predicted": predicted,
            "correct": correct,
            "cost": cost,
            "num_spans": len(sorted_spans)
        })

    # Compute metrics
    num_correct = sum(1 for d in per_document if d["correct"])
    merge_accuracy = num_correct / len(per_document)
    avg_cost = sum(d["cost"] for d in per_document) / len(per_document)

    session_end = time.time()
    latency_seconds = session_end - session_start

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"merge_accuracy: {merge_accuracy:.2%} ({num_correct}/{len(per_document)})")
    print(f"avg_cost_ratio: {avg_cost:.4f}")
    print(f"latency_seconds: {latency_seconds:.2f}")
    print(f"total_llm_calls: {total_llm_calls}")
    print(f"total_input_tokens: {total_input_tokens}")
    print(f"total_output_tokens: {total_output_tokens}")

    # Build rule info
    rules_info = []
    for rule in rules:
        avg_rule_cost = sum(rule["costs"]) / len(rule["costs"]) if rule["costs"] else 0
        rules_info.append({
            "rule_name": rule["name"],
            "description": rule["fn"].__doc__ or "",
            "coverage": rule["hits"],
            "avg_cost_ratio": round(avg_rule_cost, 6),
            "file": rule["file"]
        })
        print(f"\nRule: {rule['name']}")
        print(f"  coverage: {rule['hits']}/{len(per_document)}")
        print(f"  avg_cost_ratio: {avg_rule_cost:.4f}")

    # Save JSON log
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "question": QUESTION,
        "question_slug": QUESTION_SLUG,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "latency_seconds": round(latency_seconds, 2),
        "agent_input_tokens": total_input_tokens,
        "agent_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "num_rules": len(rules),
        "merge_accuracy": round(merge_accuracy, 4),
        "avg_cost_ratio": round(avg_cost, 6),
        "rules": rules_info,
        "per_document": per_document
    }

    log_path = RULES_DIR / f"{QUESTION_SLUG}_claude-opus-4-5_{timestamp}_rule_gen.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nLog saved to: {log_path}")

    return log_data

if __name__ == "__main__":
    main()
