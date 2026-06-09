"""Cost-Descent rule application strategy: halving-prune over cost-sorted rules.

Given a refined rule set (from refinement, or returned directly by an agent in the
rule-end-to-end strategy), this strategy hands the expensive answer model (gpt54) the
*smallest still-sufficient* retrieval instead of the full refined merge:

  1. Sort the refined rules by per-doc cost ascending (cost = tokens that rule retrieves
     on THIS document). `top-k` = the k cheapest rules; their merges are nested.
  2. Retrieve `top-n` (all refined rules). Ask gpt54mini if it contains the answer.
       - NO  -> fall back to the full Step-2 generation pool, answer with gpt54.
                (identical to the Default strategy's miss-path)
       - YES -> descend.
  3. Descend: test `top-n/2`. If gpt54mini still says YES, accept it and recurse to
     `top-n/4`, ... down to `top-1`. The first NO stops the descent.
  4. Answer once with gpt54 on the smallest passing subset (the last YES level).

The expensive model is called exactly once. Up to 1 + floor(log2 n) cheap gate calls.

Token accounting: the per-doc log records BOTH gpt54mini (summed across gate levels) and
gpt54 tokens, but the reported cost ratio uses gpt54 tokens only (gpt54mini is logged but
excluded from the cost metric).

See docs/approach/rule_apply_descent.md for the design and analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse retrieval + LLM helpers; no need to re-implement prompts or token plumbing.
from rule_refine.selection.select_rules_pareto_proxy import _retrieve_merge
from rule_apply.default import relevance_check, qa_call


# ── Cost ordering (per-doc retrieved tokens) ──────────────────────────────────

def _sort_rules_by_cost(
    document:    dict,
    rule_names:  list[str],
    rule_folder: Path,
) -> list[tuple[str, int]]:
    """Order rules by their per-doc retrieved-token count, ascending.

    A rule's cost on this document is the token count of what it retrieves here
    (via the same merge primitive, applied to that single rule). Ties broken by
    rule name for deterministic ordering. Returns [(rule_name, cost_tokens), ...].
    """
    costs: list[tuple[str, int]] = []
    for rn in rule_names:
        _, tok = _retrieve_merge(document, [rn], rule_folder)
        costs.append((rn, tok))
    costs.sort(key=lambda rc: (rc[1], rc[0]))
    return costs


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_with_descent(
    document:        dict,
    question:        str,
    refined_rules:   list[str],
    all_rules:       list[str],
    rule_folder:     Path,
    fallback_folder: Path | None = None,
    relevance_model: str = "gpt54mini",
    qa_model:        str = "gpt54",
) -> dict[str, Any]:
    """Apply refined rules via cost-descent halving with gpt54mini gating.

    `refined_rules` live in `rule_folder`; the fallback `all_rules` live in
    `fallback_folder` (defaults to `rule_folder` when the two coincide, e.g. the
    agent rule-end-to-end case where the applied set is also the full pool).

    Returns a dict with:
      predicted, retrieved_text, retrieved_tokens_used (gpt54 context size = cost),
      fallback_triggered, final_k, final_n, descent_trace (per-level k/tokens/verdict),
      tokens={mini_in, mini_out, gpt54_in, gpt54_out} (mini summed across gate levels).
    The cost ratio the caller reports = retrieved_tokens_used / doc_tokens (gpt54 only).
    """
    fallback_folder = fallback_folder or rule_folder
    # Cost-sort the refined rules (cheapest first) -> r1..rn
    cost_order = _sort_rules_by_cost(document, refined_rules, rule_folder)
    sorted_rules = [rn for rn, _ in cost_order]
    n = len(sorted_rules)

    mini_in_total = mini_out_total = 0
    descent_trace: list[dict[str, Any]] = []

    def _gate(k: int) -> tuple[str, int, bool]:
        """Retrieve top-k, run the gpt54mini gate, accumulate mini tokens, trace it."""
        nonlocal mini_in_total, mini_out_total
        text, tok = _retrieve_merge(document, sorted_rules[:k], rule_folder)
        has, m_in, m_out = relevance_check(text, question, model_name=relevance_model)
        mini_in_total  += m_in
        mini_out_total += m_out
        descent_trace.append({"k": k, "tokens": tok, "verdict": "yes" if has else "no"})
        return text, tok, has

    # ── Gate 0: is the answer in the full refined set (top-n)? ──
    if n == 0:
        has_top_n = False
        text_prev, tokens_prev = "", 0
    else:
        text_prev, tokens_prev, has_top_n = _gate(n)

    # ── Fallback path: refined set insufficient -> full generation pool ──
    if not has_top_n:
        retrieved_used, tokens_used = _retrieve_merge(document, all_rules, fallback_folder)
        predicted, qa_in, qa_out = qa_call(retrieved_used, question, model_name=qa_model)
        return {
            "predicted":             predicted,
            "retrieved_text":        retrieved_used,
            "retrieved_tokens_used": tokens_used,
            "fallback_triggered":    True,
            "final_k":               0,
            "final_n":               n,
            "descent_trace":         descent_trace,
            "tokens": {
                "mini_in":   mini_in_total,
                "mini_out":  mini_out_total,
                "gpt54_in":  qa_in,
                "gpt54_out": qa_out,
            },
        }

    # ── Descend: halve while the gate keeps saying YES ──
    final_k = n
    k = n
    while k // 2 >= 1:
        k_next = k // 2
        text_next, tokens_next, has_next = _gate(k_next)
        if has_next:
            text_prev, tokens_prev, final_k, k = text_next, tokens_next, k_next, k_next
        else:
            break

    # ── Answer once with gpt54 on the smallest passing subset ──
    predicted, qa_in, qa_out = qa_call(text_prev, question, model_name=qa_model)

    return {
        "predicted":             predicted,
        "retrieved_text":        text_prev,
        "retrieved_tokens_used": tokens_prev,
        "fallback_triggered":    False,
        "final_k":               final_k,
        "final_n":               n,
        "descent_trace":         descent_trace,
        "tokens": {
            "mini_in":   mini_in_total,
            "mini_out":  mini_out_total,
            "gpt54_in":  qa_in,
            "gpt54_out": qa_out,
        },
    }
