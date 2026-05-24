#!/usr/bin/env python3
"""Test the LLM judge directly."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import gpt54

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

Reply with exactly one word: CORRECT or INCORRECT"""

def judge_answer(question, predicted, ground_truth):
    """Use LLM to judge if predicted answer is correct."""
    prompt = f"""Question: {question}

Predicted answer: {predicted}

Ground truth answer: {ground_truth}

Is the predicted answer correct?"""

    response = gpt54.chat_completions(
        prompt=prompt,
        system=JUDGE_SYSTEM,
        max_completion_tokens=10,
        temperature=0.0
    )
    print(f"Judge response: '{response}'")
    return "CORRECT" in response.upper()

# Test cases
QUESTION = "What is long-term debt at year-end (0 if none)?"

test_cases = [
    ("$31,816 million", "$101,406 million"),  # Amazon - should be INCORRECT
    ("$3,605 million", "$3.605 billion"),     # Should be CORRECT (equivalent)
    ("$6,577 million", "6487"),                # Costco - should be INCORRECT
]

for predicted, ground_truth in test_cases:
    print(f"\nPredicted: {predicted}")
    print(f"Ground truth: {ground_truth}")
    result = judge_answer(QUESTION, predicted, ground_truth)
    print(f"Result: {'CORRECT' if result else 'INCORRECT'}")
