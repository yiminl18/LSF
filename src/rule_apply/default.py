"""Default rule application strategy: refined rules with full-pool fallback.

When applying a refined rule set to an unsampled document:
  1. Retrieve text using the refined rules (no LLM).
  2. Ask gpt54mini whether the retrieved text contains an answer to the question.
  3. If YES, use gpt54 to extract the answer from the refined retrieval.
  4. If NO, fall back to the full rule pool, then use gpt54 to extract.

Returns a per-doc dict with prediction, fallback flag, relevance verdict, and
token accounting for both models. The caller is responsible for judging the
prediction against ground truth.

See docs/rule_apply_with_fallback.md for the design rationale.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse the pure-Python retrieval helper from the proxy variant
from rule_refine.selection.select_rules_pareto_proxy import _retrieve_merge


# ── Prompts ──────────────────────────────────────────────────────────────────

_RELEVANCE_SYSTEM = """You are a relevance judge for a financial document QA system.
Given a passage extracted from a financial filing and a question, decide if the passage
contains enough information to answer the question correctly.

Reply with exactly one word: YES or NO.

Rules:
- YES iff the passage contains the specific value, name, or fact the question asks for.
- NO if the passage is missing the answer, contains only partial information, or contains unrelated text.
- Do not guess. If unsure, reply NO."""

_QA_SYSTEM = (
    "You are a financial document QA assistant.\n"
    "You are given a passage extracted from a financial filing and a question.\n"
    "Answer the question using only the provided passage.\n"
    'If the passage does not contain enough information to answer, reply with "NOT FOUND".\n'
    "Return only the answer — a short value or phrase, not a full sentence."
)


# ── LLM helpers ──────────────────────────────────────────────────────────────

def relevance_check(retrieved_text: str, question: str, model_name: str = "gpt54mini") -> tuple[bool, int, int]:
    """Cheap yes/no relevance check. Returns (has_answer, in_tokens, out_tokens)."""
    if not retrieved_text or not retrieved_text.strip():
        return False, 0, 0
    model_mod = importlib.import_module(f"models.{model_name}")
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {"role": "user",   "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
            ],
            max_completion_tokens=10,
            temperature=0.0,
        )
    except Exception as e:
        if "content_filter" in str(e) or "content management" in str(e):
            warnings.warn(f"Content filter on relevance check; defaulting to NO (fallback)")
            return False, 0, 0
        raise

    verdict = (resp.choices[0].message.content or "").strip().lower()
    has = verdict.startswith("yes")
    in_tok  = resp.usage.prompt_tokens     if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return has, in_tok, out_tok


def qa_call(retrieved_text: str, question: str, model_name: str = "gpt54") -> tuple[str | None, int, int]:
    """QA call using production model. Returns (predicted, in_tokens, out_tokens)."""
    model_mod = importlib.import_module(f"models.{model_name}")
    try:
        resp = model_mod.client.chat.completions.create(
            model=model_mod.AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user",   "content": f"Passage:\n{retrieved_text}\n\nQuestion: {question}"},
            ],
            max_completion_tokens=500,
            temperature=0.0,
        )
    except Exception as e:
        if "content_filter" in str(e) or "content management" in str(e):
            warnings.warn(f"Content filter on QA; returning None")
            return None, 0, 0
        raise

    pred = (resp.choices[0].message.content or "").strip() or None
    in_tok  = resp.usage.prompt_tokens     if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    return pred, in_tok, out_tok


# ── Main entry point ─────────────────────────────────────────────────────────

def apply_with_fallback(
    document:        dict,
    question:        str,
    refined_rules:   list[str],
    all_rules:       list[str],
    rule_folder:     Path,
    relevance_model: str = "gpt54mini",
    qa_model:        str = "gpt54",
) -> dict[str, Any]:
    """Apply refined rules with gpt54mini gating and full-pool fallback.

    Returns a dict with:
      predicted, retrieved_text, retrieved_tokens_refined, retrieved_tokens_used,
      fallback_triggered, relevance_verdict, tokens={mini_in, mini_out, gpt54_in, gpt54_out}.
    """
    # Step 1: retrieve with refined rules (no LLM)
    retrieved_S, tokens_S = _retrieve_merge(document, refined_rules, rule_folder)

    # Step 2: relevance check with gpt54mini
    has_answer, rel_in, rel_out = relevance_check(retrieved_S, question, model_name=relevance_model)
    fallback_triggered = not has_answer

    # Step 3: select retrieval (refined vs full pool)
    if has_answer:
        retrieved_used, tokens_used = retrieved_S, tokens_S
        tokens_R = 0
    else:
        retrieved_used, tokens_used = _retrieve_merge(document, all_rules, rule_folder)
        tokens_R = tokens_used

    # Step 4: QA with gpt54
    predicted, qa_in, qa_out = qa_call(retrieved_used, question, model_name=qa_model)

    return {
        "predicted":                 predicted,
        "retrieved_text":            retrieved_used,
        "retrieved_tokens_refined":  tokens_S,
        "retrieved_tokens_full":     tokens_R,
        "retrieved_tokens_used":     tokens_used,
        "fallback_triggered":        fallback_triggered,
        "relevance_verdict":         "yes" if has_answer else "no",
        "tokens": {
            "mini_in":   rel_in,
            "mini_out":  rel_out,
            "gpt54_in":  qa_in,
            "gpt54_out": qa_out,
        },
    }
