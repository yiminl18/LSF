#!/usr/bin/env python3
"""
Full LLM-based evaluation for trading symbol rules.
Computes merge_accuracy using LLM QA and LLM judge.
"""

import json
import os
import sys
import time
import tiktoken
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import Azure OpenAI model
try:
    from models import gpt54 as model_mod
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("Warning: model module not found. Using mock evaluation.")

SAMPLED_DOCS = [
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
    'Pfizer_2023Q2_10Q',
    'ACTIVSIONBLIZZARD_2023Q2_10Q',
    '3M_2023Q2_10Q',
    'AMCOR_2022_8K_2022-07-01',
    'COSTCO_2023_8K_dated-2023-08-09',
    'COSTCO_2023_8K_dated-2023-08-16',
    'MGMRESORTS_2023_8K_dated-2023-03-01',
    'FOOTLOCKER_2022_8K_dated-2022-05-20'
]

QUESTION = "What is/are the trading symbol(s) and listing exchange(s)?"

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def load_ground_truth():
    with open('data/financebench/sample_mix_doc_labels.json', 'r') as f:
        labels = json.load(f)

    ground_truth = {}
    for doc_name in SAMPLED_DOCS:
        pdf_name = doc_name + ".pdf"
        if pdf_name in labels and QUESTION in labels[pdf_name]:
            ans = labels[pdf_name][QUESTION]
            if isinstance(ans, list):
                parts = []
                for item in ans:
                    if isinstance(item, dict):
                        sym = item.get('symbol') or item.get('Trading Symbol')
                        exch = item.get('exchange') or item.get('Exchange')
                        if sym and exch:
                            parts.append(f"{sym}, {exch}")
                ground_truth[doc_name] = "; ".join(parts) if parts else str(ans)
            else:
                ground_truth[doc_name] = str(ans)
    return ground_truth

def load_doc(doc_name):
    path = f'data/financebench/processing/{doc_name}_reconstructed.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def rule_securities_12b_with_symbol(doc: dict) -> list[dict]:
    """Match Section 12(b) securities info AND Item 5 Market section symbol info."""
    try:
        texts = doc.get('texts', [])
        result = []
        seen = set()

        # Part 1: Section 12(b) on page 1-2
        in_section = False
        for i, s in enumerate(texts):
            page = s.get('page_no', 0)
            if page > 2:
                continue
            text = s.get('text', '').lower()
            if 'securities registered' in text and '12(b)' in text:
                in_section = True
            if in_section:
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))
                if 'indicate by check mark' in text:
                    break

        # Part 2: Page 1-2 exchange name spans (for documents with out-of-order spans)
        exchange_names = ['new york stock exchange', 'nasdaq', 'nyse', 'chicago stock exchange']
        for s in texts:
            page = s.get('page_no', 0)
            if page > 2:
                continue
            text = s.get('text', '').lower()
            if any(ex in text for ex in exchange_names):
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))

        # Part 3: Item 5 Market section spans (for older filings like Boeing, Costco 2018)
        for s in texts:
            text = s.get('text', '').lower()
            path = s.get('structure', {}).get('path_text', '').lower()
            if ('item 5' in path or 'market for registrant' in path):
                if ('trades under' in text and 'symbol' in text) or \
                   ('traded on' in text and 'symbol' in text):
                    if id(s) not in seen:
                        result.append(s)
                        seen.add(id(s))

        return result
    except Exception:
        return []

def call_llm_qa(passage: str, question: str) -> tuple[str, int, int]:
    """Call LLM to answer question based on passage. Returns (answer, input_tokens, output_tokens)."""
    if not HAS_MODEL:
        return "MOCK_ANSWER", 0, 0

    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence.\n"
        "For trading symbols: list ALL trading symbols registered under Section 12(b), including notes and other securities, in format: SYMBOL — Exchange; SYMBOL2 — Exchange2"
    )
    user_prompt = f"Passage:\n{passage}\n\nQuestion: {question}"

    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=200,
        temperature=0.0,
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return answer, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0

def call_llm_judge(question: str, predicted: str, ground_truth: str) -> tuple[bool, int, int]:
    """Call LLM to judge if predicted answer is correct. Returns (is_correct, input_tokens, output_tokens)."""
    if not HAS_MODEL:
        # Fallback substring check
        pred_lower = predicted.lower()
        gt_lower = ground_truth.lower()
        parts = gt_lower.replace(",", " ").replace("—", " ").replace("-", " ").split()
        key_parts = [p for p in parts if len(p) > 2]
        hits = sum(1 for p in key_parts if p in pred_lower)
        return hits >= len(key_parts) * 0.5, 0, 0

    judge_system = """\
You are an answer equivalence judge for a financial document QA system.
You will be given a question, a predicted answer, and a ground truth answer.
Judge whether the predicted answer is correct — meaning semantically equivalent
to the ground truth, ignoring minor formatting differences.

Equivalence rules:
- Treat "2017" and "year 2017" as the same
- Treat "$4.5 billion" and "4,500 million" as the same if numerically equal
- Treat "NYSE" and "New York Stock Exchange" as the same
- Treat "NASDAQ" and "Nasdaq Global Select Market" as equivalent for exchange name
- Ignore leading/trailing whitespace, punctuation, and capitalization differences
- If the predicted answer is "NOT FOUND" or null, always judge as incorrect
- For multiple trading symbols, all symbols must be present

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )

    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": judge_system},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )

    judge_raw = (response.choices[0].message.content or "").strip()
    usage = response.usage

    verdict = judge_raw.strip().lower()
    is_correct = verdict == "correct"

    return is_correct, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def evaluate():
    """Run full evaluation."""
    start_time = time.time()

    ground_truth = load_ground_truth()

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    correct_count = 0
    total_count = 0
    total_cost = 0.0

    print(f"Evaluating {len(SAMPLED_DOCS)} documents...")
    print("=" * 60)

    for doc_name in SAMPLED_DOCS:
        doc = load_doc(doc_name)
        if doc is None:
            print(f"  {doc_name}: SKIPPED (file not found)")
            continue

        if doc_name not in ground_truth:
            print(f"  {doc_name}: SKIPPED (no ground truth)")
            continue

        gt_answer = ground_truth[doc_name]

        # Apply rule
        spans = rule_securities_12b_with_symbol(doc)
        retrieved_text = "\n\n".join(s.get("text", "") for s in spans)

        # Calculate cost
        full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
        full_tokens = count_tokens(full_text)
        retrieved_tokens = count_tokens(retrieved_text)
        cost_ratio = retrieved_tokens / full_tokens if full_tokens > 0 else 0

        # Call LLM for QA
        predicted_answer, qa_in, qa_out = call_llm_qa(retrieved_text, QUESTION)
        total_input_tokens += qa_in
        total_output_tokens += qa_out
        total_llm_calls += 1

        # Call LLM judge
        is_correct, judge_in, judge_out = call_llm_judge(QUESTION, predicted_answer, gt_answer)
        total_input_tokens += judge_in
        total_output_tokens += judge_out
        total_llm_calls += 1

        total_count += 1
        total_cost += cost_ratio
        if is_correct:
            correct_count += 1

        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"  {doc_name}: {status}")
        print(f"    Ground truth: {gt_answer}")
        print(f"    Predicted:    {predicted_answer}")
        print(f"    Cost ratio:   {cost_ratio:.4f}")
        print()

        results.append({
            "doc_name": doc_name,
            "ground_truth": gt_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "cost_ratio": cost_ratio,
            "num_spans": len(spans),
            "retrieved_tokens": retrieved_tokens,
            "full_doc_tokens": full_tokens,
        })

    latency = time.time() - start_time
    merge_accuracy = correct_count / total_count if total_count > 0 else 0
    avg_cost = total_cost / total_count if total_count > 0 else 0

    print("=" * 60)
    print(f"Summary:")
    print(f"  Merge Accuracy: {merge_accuracy:.2%} ({correct_count}/{total_count})")
    print(f"  Avg Cost Ratio: {avg_cost:.4f}")
    print(f"  Total LLM Calls: {total_llm_calls}")
    print(f"  Total Input Tokens: {total_input_tokens}")
    print(f"  Total Output Tokens: {total_output_tokens}")
    print(f"  Latency: {latency:.2f}s")

    return {
        "merge_accuracy": merge_accuracy,
        "avg_cost_ratio": avg_cost,
        "correct_count": correct_count,
        "total_count": total_count,
        "latency_seconds": latency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_llm_calls": total_llm_calls,
        "per_document": results,
    }


if __name__ == "__main__":
    result = evaluate()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"local/trading_symbol_eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
