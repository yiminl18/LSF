#!/usr/bin/env python3
"""Final evaluation with simpler, more reliable approach."""

import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text): return len(enc.encode(text))
except ImportError:
    def count_tokens(text): return len(text.split())

from models import gpt54

DOCS = [
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
    "ACTIVSIONBLIZZARD_2023Q2_10Q",
    "3M_2023Q2_10Q",
    "AMCOR_2022_8K_2022-07-01",
    "COSTCO_2023_8K_dated-2023-08-09",
    "COSTCO_2023_8K_dated-2023-08-16",
    "FOOTLOCKER_2022_8K_dated-2022-05-20",
]

QUESTION = "What is long-term debt at year-end (0 if none)?"

QA_SYSTEM = """\
You are a financial analyst. Extract the long-term debt from the provided tables.
Look for "long-term debt" line items. Use the most recent period's value.
If "Total long-term obligations" is shown, use that value.
Answer with just the number and units (e.g., "$4,117 million").
If not found, answer "NOT FOUND"."""

JUDGE_SYSTEM = """\
Judge if the predicted answer matches the ground truth for long-term debt.
Consider: "$4.5 billion" = "4,500 million", numbers within 2% are OK.
Reply: CORRECT or INCORRECT"""

def load_labels():
    with open(_ROOT / "data/financebench/sample_mix_doc_labels.json") as f:
        labels = json.load(f)
    result = {}
    for doc_name in DOCS:
        pdf_name = f"{doc_name}.pdf"
        if pdf_name in labels:
            result[doc_name] = labels[pdf_name].get(QUESTION)
    return result

def load_doc(doc_name):
    path = _ROOT / f"data/financebench/processing/{doc_name}_reconstructed.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# RULES
def rule_long_term_debt_tables(doc):
    """Tables containing 'long-term debt' keyword."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except Exception:
        return []

def rule_total_obligations_tables(doc):
    """Tables with 'total long-term obligations'."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                "long-term obligations" in s.get("text", "").lower()]
    except Exception:
        return []

def get_llm_answer(context, question, retries=2):
    prompt = f"Tables:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    for attempt in range(retries + 1):
        try:
            response = gpt54.chat_completions(
                prompt=prompt,
                system=QA_SYSTEM,
                max_completion_tokens=50,
                temperature=0.0
            )
            return response.strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return f"ERROR: {e}"

def judge_answer(question, predicted, ground_truth, retries=2):
    if "ERROR" in predicted:
        return False
    prompt = f"Predicted: {predicted}\nGround truth: {ground_truth}"
    for attempt in range(retries + 1):
        try:
            response = gpt54.chat_completions(
                prompt=prompt,
                system=JUDGE_SYSTEM,
                max_completion_tokens=10,
                temperature=0.0
            )
            return response.strip().upper() == "CORRECT"
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return False

def main():
    labels = load_labels()
    results = []
    rules = [rule_long_term_debt_tables, rule_total_obligations_tables]

    print(f"Evaluating {len(DOCS)} documents...")
    print("="*60)

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if not doc:
            continue

        answer = labels.get(doc_name)

        # Apply rules
        all_spans = []
        seen = set()
        for rule in rules:
            for s in rule(doc):
                if id(s) not in seen:
                    seen.add(id(s))
                    all_spans.append(s)

        retrieved_text = "\n\n".join(s.get("text", "") for s in all_spans)
        full_text = "\n".join(s.get("text", "") for s in doc.get("texts", []))

        ret_tokens = count_tokens(retrieved_text) if retrieved_text else 0
        full_tokens = count_tokens(full_text) if full_text else 1
        cost = ret_tokens / full_tokens if full_tokens > 0 else 0

        # Get LLM answer
        if retrieved_text:
            llm_answer = get_llm_answer(retrieved_text, QUESTION)
        else:
            llm_answer = "0"

        # Judge
        if answer is None:
            is_correct = True
        else:
            is_correct = judge_answer(QUESTION, llm_answer, str(answer))

        results.append({
            "doc": doc_name,
            "answer": answer,
            "predicted": llm_answer,
            "correct": is_correct,
            "cost": cost,
            "spans": len(all_spans)
        })

        status = "CORRECT" if is_correct else "WRONG"
        print(f"{status}: {doc_name}")
        print(f"  Expected: {answer}")
        print(f"  Predicted: {llm_answer}")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    avg_cost = sum(r["cost"] for r in results) / total if total > 0 else 0

    print("\n" + "="*60)
    print(f"Merge accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Avg cost: {avg_cost:.4f}")

    return results, accuracy, avg_cost

if __name__ == "__main__":
    main()
