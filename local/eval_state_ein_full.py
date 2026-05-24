#!/usr/bin/env python3
"""Full evaluation of state/EIN rules with LLM judge."""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tiktoken

DOCS = [
    'AMCOR_2019_10K', 'COSTCO_2017_10K', 'BOEING_2018_10K', 'AMAZON_2018_10K',
    'EBAY_2021_10K', 'AMAZON_2016_10K', 'CORNING_2022_10K', 'NIKE_2021_10K',
    'LOCKHEEDMARTIN_2022_10K', 'JOHNSON_JOHNSON_2022_10K'
]

QUESTION = 'What is the state (or other jurisdiction) of incorporation and the IRS Employer Identification Number?'
QUESTION_SLUG = 'what_is_the_state__or_other_jurisdiction__of_incorporation_a'

START_TIME = time.time()
TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_LLM_CALLS = 0


def rule_page1_jurisdiction_ein(doc: dict) -> list[dict]:
    """Match page 1 spans containing state/jurisdiction and EIN info."""
    try:
        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_span = span.get("text_span", "")
            combined = (text + " " + text_span).lower()
            if any(kw in combined for kw in [
                "jurisdiction", "state of incorporation",
                "employer identification", "i.r.s. employer", "irs employer"
            ]):
                results.append(span)
                if i > 0:
                    prev = texts[i - 1]
                    if prev.get("page_no") == 1:
                        prev_text = prev.get("text", "")
                        if "registrant" not in prev_text.lower() and "exact name" not in prev_text.lower():
                            results.append(prev)
            if re.search(r'\b\d{2}-\d{7}\b', text):
                results.append(span)
        seen = set()
        unique = []
        for s in results:
            key = id(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
    except Exception:
        return []


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def call_llm_qa(passage: str, question: str) -> tuple[str, int, int]:
    """Call LLM to answer question from passage. Returns (answer, input_tokens, output_tokens)."""
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_LLM_CALLS

    from models import gpt54

    system_prompt = (
        "You are a financial document QA assistant.\n"
        "You are given a passage extracted from a financial filing and a question.\n"
        "Answer the question using only the provided passage.\n"
        'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
        "Return only the answer — a short value or phrase, not a full sentence."
    )
    user_prompt = f"Passage:\n{passage}\n\nQuestion: {question}"

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=200,
        temperature=0.0,
    )

    TOTAL_LLM_CALLS += 1
    usage = response.usage
    input_toks = usage.prompt_tokens if usage else 0
    output_toks = usage.completion_tokens if usage else 0
    TOTAL_INPUT_TOKENS += input_toks
    TOTAL_OUTPUT_TOKENS += output_toks

    answer = (response.choices[0].message.content or "").strip()
    return answer, input_toks, output_toks


def call_llm_judge(question: str, predicted: str, ground_truth: str) -> tuple[bool, int, int]:
    """Call LLM judge to compare predicted vs ground truth. Returns (is_correct, input_tokens, output_tokens)."""
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_LLM_CALLS

    from models import gpt54

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

Reply with exactly one word: CORRECT or INCORRECT"""

    user_prompt = (
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Predicted: {predicted}"
    )

    response = gpt54.client.chat.completions.create(
        model=gpt54.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )

    TOTAL_LLM_CALLS += 1
    usage = response.usage
    input_toks = usage.prompt_tokens if usage else 0
    output_toks = usage.completion_tokens if usage else 0
    TOTAL_INPUT_TOKENS += input_toks
    TOTAL_OUTPUT_TOKENS += output_toks

    verdict = (response.choices[0].message.content or "").strip().lower()
    is_correct = verdict == "correct"
    return is_correct, input_toks, output_toks


def main():
    global START_TIME

    # Load labels
    with open('data/financebench/sample_doc_labels.json') as f:
        labels = json.load(f)

    enc = tiktoken.get_encoding("cl100k_base")

    results = []
    total_cost = 0
    num_correct = 0

    print(f"\nEvaluating rule_page1_jurisdiction_ein on {len(DOCS)} documents...")
    print("=" * 70)

    for doc_name in DOCS:
        # Load document
        with open(f'data/financebench/processing/{doc_name}_reconstructed.json') as f:
            doc = json.load(f)

        # Get ground truth
        gt = labels.get(f'{doc_name}.pdf', {}).get(QUESTION, '')

        # Apply rule
        retrieved_spans = rule_page1_jurisdiction_ein(doc)
        retrieved_text = "\n\n".join(s.get("text", "") for s in retrieved_spans)

        # Calculate cost
        full_text = " ".join(s.get("text", "") for s in doc.get("texts", []))
        retrieved_tokens = len(enc.encode(retrieved_text))
        full_tokens = len(enc.encode(full_text))
        cost = retrieved_tokens / full_tokens if full_tokens > 0 else 0
        total_cost += cost

        # Call LLM to get answer
        predicted, qa_in, qa_out = call_llm_qa(retrieved_text, QUESTION)

        # Call LLM judge
        is_correct, judge_in, judge_out = call_llm_judge(QUESTION, predicted, gt)

        if is_correct:
            num_correct += 1
            status = "CORRECT"
        else:
            status = "INCORRECT"

        print(f"{status:10} {doc_name}")
        print(f"           GT:   {gt}")
        print(f"           Pred: {predicted}")
        print(f"           Cost: {cost:.4f}, Spans: {len(retrieved_spans)}")
        print()

        results.append({
            'doc_name': doc_name,
            'ground_truth': gt,
            'predicted': predicted,
            'is_correct': is_correct,
            'cost': cost,
            'retrieved_tokens': retrieved_tokens,
            'num_spans': len(retrieved_spans),
        })

    latency = time.time() - START_TIME
    accuracy = num_correct / len(DOCS)
    avg_cost = total_cost / len(DOCS)

    print("=" * 70)
    print(f"SUMMARY")
    print(f"  Merge accuracy:     {accuracy:.2%} ({num_correct}/{len(DOCS)})")
    print(f"  Avg cost ratio:     {avg_cost:.6f}")
    print(f"  Latency:            {latency:.2f}s")
    print(f"  Total LLM calls:    {TOTAL_LLM_CALLS}")
    print(f"  Total input tokens: {TOTAL_INPUT_TOKENS}")
    print(f"  Total output tokens:{TOTAL_OUTPUT_TOKENS}")

    # Return summary for logging
    return {
        'accuracy': accuracy,
        'avg_cost': avg_cost,
        'latency': latency,
        'results': results,
        'total_llm_calls': TOTAL_LLM_CALLS,
        'total_input_tokens': TOTAL_INPUT_TOKENS,
        'total_output_tokens': TOTAL_OUTPUT_TOKENS,
    }


if __name__ == "__main__":
    main()
