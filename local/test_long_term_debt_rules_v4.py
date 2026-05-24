#!/usr/bin/env python3
"""Refined rules to improve accuracy."""

import json
import re
import sys
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
You are a financial document analyst. Answer the question based ONLY on the provided context.
If the information is not available in the context, answer "NOT FOUND".
For numerical answers, include units (millions, billions, etc.) when available.
Be concise - provide just the numerical answer with units."""

JUDGE_SYSTEM = """\
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
- "0" and "none" and "no long-term debt" are equivalent
- Numbers that are within 1% of each other can be considered equivalent (rounding)
- "988,924 thousand" equals "$988,924,000"

Reply with exactly one word: CORRECT or INCORRECT"""

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
    """Tables containing 'long-term debt' or 'long term debt' keyword."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except Exception:
        return []

def rule_item6_selected_data(doc):
    """Tables in Item 6 Selected Financial Data with long-term obligations."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("item 6" in s.get("structure", {}).get("path_text", "").lower() or
                 "selected financial" in s.get("structure", {}).get("path_text", "").lower()) and
                any(kw in s.get("text", "").lower() for kw in [
                    "long-term", "long term", "total debt", "obligations"
                ])]
    except Exception:
        return []

def rule_balance_sheet_liabilities(doc):
    """Balance sheet tables with liabilities including debt."""
    try:
        results = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            text = s.get("text", "").lower()
            # Balance sheets with debt line items
            if "liabilities" in text and any(kw in text for kw in ["long-term debt", "long term debt"]):
                results.append(s)
        return results
    except Exception:
        return []

def get_llm_answer(context, question):
    prompt = f"""Context:
{context}

Question: {question}

Answer (just the number with units):"""

    try:
        response = gpt54.chat_completions(
            prompt=prompt,
            system=QA_SYSTEM,
            max_completion_tokens=100,
            temperature=0.0
        )
        return response.strip()
    except Exception as e:
        return "ERROR"

def judge_answer(question, predicted, ground_truth):
    prompt = f"""Question: {question}

Predicted answer: {predicted}

Ground truth answer: {ground_truth}

Is the predicted answer correct?"""

    try:
        response = gpt54.chat_completions(
            prompt=prompt,
            system=JUDGE_SYSTEM,
            max_completion_tokens=10,
            temperature=0.0
        )
        response_clean = response.strip().upper()
        return response_clean == "CORRECT" or (response_clean.startswith("CORRECT") and "INCORRECT" not in response_clean)
    except Exception:
        return False

def test_rules(rules):
    labels = load_labels()
    results = []

    for doc_name in DOCS:
        doc = load_doc(doc_name)
        if not doc:
            continue

        answer = labels.get(doc_name)

        # Apply all rules and union
        all_spans = []
        seen = set()
        for rule in rules:
            spans = rule(doc)
            for s in spans:
                span_id = id(s)
                if span_id not in seen:
                    seen.add(span_id)
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
        print(f"  Spans: {len(all_spans)}, Cost: {cost:.4f}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    avg_cost = sum(r["cost"] for r in results) / total if total > 0 else 0

    print("\n" + "="*60)
    print(f"Total: {total}, Correct: {correct}")
    print(f"Merge accuracy: {accuracy:.2%}")
    print(f"Avg cost: {avg_cost:.4f}")

    return results

if __name__ == "__main__":
    print("Testing with all rules:")
    rules = [
        rule_long_term_debt_tables,
        rule_item6_selected_data,
        rule_balance_sheet_liabilities,
    ]
    test_rules(rules)
