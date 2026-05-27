"""Pareto-frontier rule selection v3: v2 + cumulative-prefix Phase B' fallback.

Extends the original Pareto pipeline (cost-effectiveness sort + greedy cover)
with two additional phases:

  Phase B — accuracy floor: after greedy completes, eval merge_acc against
  baseline. If below, admit the next-best rule by cost-effectiveness order
  and re-eval. Repeat until merge_acc >= base or pool exhausted.

  Phase C — backward pruning: for each rule in S (last-added first), test
  removal. Drop the rule iff merge_acc remains >= base. Catches retrieval-
  noise non-monotonicity (a rule whose addition gained one doc but lost
  another).

Together these phases recover v1's accuracy guarantee while keeping the
cost-effectiveness sort that Pareto provides.

This module is additive — does not modify select_rules.py, select_rules_pareto.py,
or any earlier variant.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_apply.merge                     import rule_apply_merge
from rule_refine.selection.eval_judge           import judge
from rule_refine.selection.baseline_targets     import load_target_docs
from rule_refine.selection.coverage_check       import load_or_compute_coverage
from rule_refine.selection.select_rules         import _greedy_cover
from rule_refine.selection.select_rules_pareto  import (
    _sort_by_cost_effectiveness,
    _build_frontier,
    make_query_table,
    make_knee_point,
)

_DEFAULT_OUTPUT_DIR = (
    "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selector_run_pareto_v3"
)


# ── Merge-eval helper (full doc set) ──────────────────────────────────────────

def _eval_merge_on_docs(
    rule_names:    list[str],
    documents:     dict[str, dict],
    doc_names:     list[str],
    question:      str,
    question_slug: str,
    rules_dir:     str,
    labels:        dict,
    model_name:    str,
    output_dir:    str,
) -> tuple[float, set[str], int, int, int, int]:
    """Apply the merged rule set + judge on each doc. Return
    (accuracy, correct_doc_set, qa_calls, judge_calls, in_tokens, out_tokens)."""
    correct: set[str] = set()
    qa_calls    = 0
    judge_calls = 0
    in_tokens   = 0
    out_tokens  = 0

    for d in doc_names:
        gt = labels.get(d + ".pdf", {}).get(question)
        try:
            res = rule_apply_merge(
                document=documents[d],
                rule_names=rule_names,
                question_slug=question_slug,
                question=question,
                model_name=model_name,
                rules_dir=rules_dir,
                output_dir=output_dir,
            )
            qa_calls  += 1
            in_tokens  += res.get("input_tokens", 0)
            out_tokens += res.get("output_tokens", 0)
        except Exception as exc:
            print(f"    ERR rule_apply on {d}: {exc}")
            continue

        try:
            ok, j_in, j_out = judge(question, gt, res["predicted_answer"], model_name=model_name)
            judge_calls += 1
            in_tokens   += j_in
            out_tokens  += j_out
            if ok:
                correct.add(d)
        except Exception as exc:
            print(f"    ERR judge on {d}: {exc}")

    acc = len(correct) / len(doc_names) if doc_names else 0.0
    return acc, correct, qa_calls, judge_calls, in_tokens, out_tokens


# ── Main entry point ──────────────────────────────────────────────────────────

def run_selection_pareto_v3(
    rules_dir:           str,
    eval_merge_path:     Path,
    eval_individual_dir: Path,
    documents:           dict[str, dict],
    question_slug:       str,
    question:            str,
    labels:              dict,
    cost_profile:        dict[str, dict],
    model_name:          str = "gpt54",
    output_dir:          str = _DEFAULT_OUTPUT_DIR,
    knee_lambda:         float = 10.0,
    query_thresholds:    tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 1.0),
    max_extra_rules:     int = 20,
    max_prefix_k_iters:  int = 9,
) -> dict[str, Any]:
    """Pareto selection with accuracy-floor enforcement and backward pruning."""
    eval_data         = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    baseline_accuracy = eval_data.get("accuracy", 0.0)
    target_docs       = load_target_docs(eval_merge_path)
    n_docs            = len(documents)
    doc_names         = list(documents.keys())

    print(f"  D* = {len(target_docs)} / {n_docs} docs  baseline_accuracy={baseline_accuracy:.2f}")

    if not target_docs:
        return _empty_result(question, question_slug, baseline_accuracy)

    rule_pool = list(cost_profile.keys())
    cov_map = {
        r: load_or_compute_coverage(r, eval_individual_dir / f"{r}_eval.json", lambda: 0.0)
        for r in rule_pool
    }
    rules_sorted = _sort_by_cost_effectiveness(rule_pool, cov_map, cost_profile)

    total_qa, total_judge, total_in, total_out = 0, 0, 0, 0

    # ── Phase A: cost-effectiveness greedy cover ─────────────────────────────
    print("  [A] cost-effectiveness greedy cover...")
    S, per_rule_gained, qa, jc, in_tok, out_tok = _greedy_cover(
        rules_sorted=rules_sorted,
        target_docs=target_docs,
        documents=documents,
        question_slug=question_slug,
        question=question,
        rules_dir=rules_dir,
        labels=labels,
        model_name=model_name,
        output_dir=output_dir,
    )
    total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok
    print(f"  [A] greedy selected {len(S)} rules")

    # ── Phase B: accuracy floor ──────────────────────────────────────────────
    print(f"  [B] enforcing accuracy floor (target={baseline_accuracy:.2f})...")
    acc, correct_set, qa, jc, in_tok, out_tok = _eval_merge_on_docs(
        S, documents, doc_names, question, question_slug, rules_dir, labels,
        model_name, output_dir,
    )
    total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok
    print(f"  [B] initial merge_acc={acc:.2f}")

    extras_added: list[dict] = []
    in_set = set(S)
    pool_iter = iter([r for r in rules_sorted if r not in in_set])
    prev_correct = set(correct_set)

    while acc < baseline_accuracy and len(extras_added) < max_extra_rules:
        try:
            next_r = next(pool_iter)
        except StopIteration:
            warnings.warn(f"Pool exhausted at acc={acc:.2f} < base={baseline_accuracy:.2f}")
            break
        S.append(next_r)
        in_set.add(next_r)

        acc, correct_set, qa, jc, in_tok, out_tok = _eval_merge_on_docs(
            S, documents, doc_names, question, question_slug, rules_dir, labels,
            model_name, output_dir,
        )
        total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok

        gained_now = correct_set - prev_correct
        per_rule_gained[next_r] = gained_now
        prev_correct = set(correct_set)

        extras_added.append({
            "iter":     len(extras_added) + 1,
            "added":    next_r,
            "gained":   sorted(gained_now),
            "acc_after": round(acc, 4),
        })
        print(f"  [B] + {next_r}  gained={len(gained_now)}  acc={acc:.2f}")

    if acc < baseline_accuracy:
        print(f"  [B] WARNING: floor not reached ({acc:.2f} < {baseline_accuracy:.2f})")
    else:
        print(f"  [B] floor reached: {acc:.2f} >= {baseline_accuracy:.2f}")

    # ── Phase B' (cumulative-prefix fallback) ───────────────────────────────
    phase_b_prime_iters: list[dict] = []
    if acc < baseline_accuracy:
        print(f"  [B'] Phase B hit cap without reaching base; switching to cumulative-prefix testing...")
        remaining = [r for r in rules_sorted if r not in set(S)]
        k = 1
        iters = 0
        while iters < max_prefix_k_iters and acc < baseline_accuracy:
            iters += 1
            take = min(k, len(remaining))
            if take == 0:
                break
            trial_S = S + remaining[:take]
            acc_trial, correct_trial, qa, jc, in_tok, out_tok = _eval_merge_on_docs(
                trial_S, documents, doc_names, question, question_slug, rules_dir, labels,
                model_name, output_dir,
            )
            total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok
            phase_b_prime_iters.append({
                "iter":      iters,
                "k":         take,
                "added_n":   take,
                "acc_after": round(acc_trial, 4),
            })
            print(f"  [B\'] k={take} (added {take} new rules)  acc={acc_trial:.2f}")
            if acc_trial >= baseline_accuracy:
                # Commit cumulative prefix; mark each newly-added rule's "gained"
                for r in remaining[:take]:
                    if r not in per_rule_gained:
                        per_rule_gained[r] = set()
                S = trial_S
                in_set = set(S)
                acc = acc_trial
                correct_set = correct_trial
                # Record cumulative gains on the last batch for frontier
                gained_now = correct_set - prev_correct
                if remaining[:take]:
                    per_rule_gained[remaining[take-1]] = gained_now
                prev_correct = set(correct_set)
                break
            if take >= len(remaining):
                break
            k *= 2
        if acc < baseline_accuracy:
            print(f"  [B\'] WARNING: even full pool didn't reach base ({acc:.2f} < {baseline_accuracy:.2f})")
        else:
            print(f"  [B\'] floor reached via prefix fallback")

    # ── Phase C: backward pruning ────────────────────────────────────────────
    print("  [C] backward pruning...")
    pruned: list[dict] = []
    i = len(S) - 1
    while i >= 0:
        if len(S) <= 1:
            break
        r = S[i]
        trial = S[:i] + S[i+1:]
        acc_t, correct_t, qa, jc, in_tok, out_tok = _eval_merge_on_docs(
            trial, documents, doc_names, question, question_slug, rules_dir, labels,
            model_name, output_dir,
        )
        total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok

        if acc_t >= baseline_accuracy:
            S.pop(i)
            per_rule_gained.pop(r, None)
            pruned.append({"removed": r, "acc_after": round(acc_t, 4)})
            print(f"  [C] - {r}  acc={acc_t:.2f}  removed")
            correct_set = correct_t
        else:
            print(f"  [C]   {r}  acc_if_removed={acc_t:.2f}  kept")
        i -= 1

    # ── Final eval (after pruning) ───────────────────────────────────────────
    final_acc, final_correct, qa, jc, in_tok, out_tok = _eval_merge_on_docs(
        S, documents, doc_names, question, question_slug, rules_dir, labels,
        model_name, output_dir,
    )
    total_qa += qa; total_judge += jc; total_in += in_tok; total_out += out_tok
    print(f"  [final] selected={len(S)}  acc={final_acc:.2f}  base={baseline_accuracy:.2f}")

    # ── Frontier reconstruction (best-effort using accumulated per_rule_gained) ─
    frontier    = _build_frontier(S, per_rule_gained, cov_map, cost_profile, target_docs)
    query_table = make_query_table(frontier, thresholds=query_thresholds)
    knee_point  = make_knee_point(frontier, lam=knee_lambda)

    covered       = set().union(*per_rule_gained.values()) if per_rule_gained else set()
    uncovered     = target_docs - covered
    selected_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in S)

    return {
        "question":                    question,
        "question_slug":               question_slug,
        "mode":                        "pareto_v3",
        "baseline_accuracy":           round(baseline_accuracy, 4),
        "selected_rules":              S,
        "selected_avg_cost_ratio_sum": round(selected_cost, 6),
        "covered_docs":                sorted(covered),
        "uncovered_docs":              sorted(uncovered),
        "selector_accuracy":           round(final_acc, 4),
        "final_merge_accuracy":        round(final_acc, 4),
        "matches_base":                final_acc >= baseline_accuracy - 1e-9,
        "frontier":                    frontier,
        "query_table":                 query_table,
        "knee_point":                  knee_point,
        "knee_lambda":                 knee_lambda,
        "extras_added_phase_b":        extras_added,
        "phase_b_prime_iters":         phase_b_prime_iters,
        "pruned_phase_c":              pruned,
        "phase_a_rules_count":         len(S) - len(extras_added) + len(pruned),  # post-greedy
        "phase_b_extras_count":        len(extras_added),
        "phase_b_prime_iters_count":   len(phase_b_prime_iters),
        "phase_c_pruned_count":        len(pruned),
        "llm_calls": {
            "qa":    total_qa,
            "judge": total_judge,
        },
        "token_usage": {
            "total_input_tokens":  total_in,
            "total_output_tokens": total_out,
        },
    }


def _empty_result(question: str, slug: str, base: float) -> dict[str, Any]:
    return {
        "question":                    question,
        "question_slug":               slug,
        "mode":                        "pareto_v3",
        "baseline_accuracy":           round(base, 4),
        "selected_rules":              [],
        "selected_avg_cost_ratio_sum": 0.0,
        "covered_docs":                [],
        "uncovered_docs":              [],
        "selector_accuracy":           0.0,
        "final_merge_accuracy":        0.0,
        "matches_base":                False,
        "frontier":                    [{
            "cost": 0.0, "accuracy_match": 0.0, "rules_admitted": [],
            "added_rule": None, "rule_cov": None, "rule_avg_cost": None,
        }],
        "query_table":                 {},
        "knee_point":                  None,
        "knee_lambda":                 10.0,
        "extras_added_phase_b":        [],
        "phase_b_prime_iters":         [],
        "pruned_phase_c":              [],
        "llm_calls":   {"qa": 0, "judge": 0},
        "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0},
    }
