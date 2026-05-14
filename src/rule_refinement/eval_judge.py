"""LLM-as-judge and cheap substring proxy for answer equivalence.

Extracted from test/run_eval_merge_sampled.py so it can be shared by the
selector pipeline and the existing eval scripts.
"""

from __future__ import annotations

import importlib
import json
import warnings

_JUDGE_SYSTEM = """\
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


def judge(
    question: str,
    ground_truth,
    predicted,
    model_name: str = "gpt54",
) -> tuple[bool, int, int]:
    """Return (correct, input_tokens, output_tokens).

    correct is True iff predicted is semantically equivalent to ground_truth.
    """
    if ground_truth is None:
        return False, 0, 0
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    pred_str = str(predicted) if predicted is not None else "null"
    user_msg = (
        f"Question: {question}\n"
        f"Ground Truth: {gt_str}\n"
        f"Predicted: {pred_str}"
    )
    model_mod = importlib.import_module(f"models.{model_name}")
    response = model_mod.client.chat.completions.create(
        model=model_mod.AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=10,
        temperature=0.0,
    )
    verdict = (response.choices[0].message.content or "").strip().lower()
    if verdict not in ("correct", "incorrect"):
        warnings.warn(f"Unexpected judge response: '{verdict}'")
    usage = response.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    return verdict == "correct", in_tok, out_tok


def proxy_judge(ground_truth, retrieved_text: str) -> bool:
    """Return True iff ground_truth appears (case-insensitive) in retrieved_text.

    Used as a cheap pre-filter before the LLM judge: if the ground-truth string
    is absent from the retrieved text, the model cannot have found it, so we
    skip the judge call and treat the doc as not covered.
    """
    if ground_truth is None or not retrieved_text:
        return False
    gt_str = json.dumps(ground_truth) if not isinstance(ground_truth, str) else ground_truth
    return gt_str.lower() in retrieved_text.lower()
