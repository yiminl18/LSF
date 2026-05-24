"""Pareto-frontier rule selection.

Replaces the cost-ascending sort key with cost-effectiveness `cov(r) / W_r`
descending, then runs the existing `_greedy_cover` unchanged and records a
frontier breakpoint after every admission. The output is the full cost-vs-
accuracy curve plus a `query_table` and `knee_point` summary — not a single
selected set.

See docs/rule_selection_pareto_implementation.md for the full spec.

This module is additive: it does not modify `select_rules.py` or
`select_rules_auto_tighten.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_refinement.select_rules      import _greedy_cover
from rule_refinement.baseline_targets  import load_target_docs
from rule_refinement.coverage_check    import load_or_compute_coverage, filter_by_tau

_DEFAULT_OUTPUT_DIR = (
    "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selector_run_pareto"
)


# ── Priority (cost-effectiveness) ─────────────────────────────────────────────

def _build_cov_map(
    rule_names: list[str],
    eval_individual_dir: Path,
) -> dict[str, float]:
    """Look up cov(r) from eval_individual/<r>_eval.json for every rule in the pool."""
    return {
        r: load_or_compute_coverage(
            r,
            eval_individual_dir / f"{r}_eval.json",
            lambda: 0.0,
        )
        for r in rule_names
    }


def _sort_by_cost_effectiveness(
    rule_names: list[str],
    cov_map:    dict[str, float],
    cost_profile: dict[str, dict],
    eps: float = 1e-9,
) -> list[str]:
    """Sort by cov / avg_cost_ratio descending.

    Ties broken by cov descending, then avg_cost_ratio ascending — favours
    broad-and-cheap over narrow-and-cheap at the boundary (per §10 of the doc).
    """
    def priority(r: str) -> tuple[float, float, float]:
        cov  = cov_map.get(r, 0.0)
        cost = cost_profile[r].get("avg_cost_ratio", 0.0)
        ratio = cov / (cost + eps)
        # Sort descending on ratio and cov, ascending on cost
        return (-ratio, -cov, cost)

    return sorted(rule_names, key=priority)


# ── Frontier reconstruction ──────────────────────────────────────────────────

def _build_frontier(
    selected_order:  list[str],
    per_rule_gained: dict[str, set[str]],
    cov_map:         dict[str, float],
    cost_profile:    dict[str, dict],
    target_docs:     set[str],
) -> list[dict]:
    """Walk the admission order and emit one breakpoint per admitted rule."""
    n_target = max(1, len(target_docs))
    frontier: list[dict] = [{
        "cost":             0.0,
        "accuracy_match":   0.0,
        "rules_admitted":   [],
        "added_rule":       None,
        "rule_cov":         None,
        "rule_avg_cost":    None,
    }]
    running_cost = 0.0
    running_covered: set[str] = set()
    admitted_so_far: list[str] = []
    for r in selected_order:
        running_cost   += cost_profile[r].get("avg_cost_ratio", 0.0)
        running_covered |= per_rule_gained.get(r, set())
        admitted_so_far.append(r)
        frontier.append({
            "cost":             round(running_cost, 6),
            "accuracy_match":   round(len(running_covered) / n_target, 4),
            "rules_admitted":   list(admitted_so_far),
            "added_rule":       r,
            "rule_cov":         round(cov_map.get(r, 0.0), 4),
            "rule_avg_cost":    round(cost_profile[r].get("avg_cost_ratio", 0.0), 6),
        })
    return frontier


def make_query_table(
    frontier:   list[dict],
    thresholds: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 1.0),
) -> dict[str, dict | None]:
    """For each accuracy threshold, return the cheapest frontier row that hits it."""
    out: dict[str, dict | None] = {}
    for t in thresholds:
        match = next((p for p in frontier if p["accuracy_match"] >= t), None)
        out[f"{t:.2f}"] = (
            {"cost": match["cost"], "n_rules": len(match["rules_admitted"])}
            if match else None
        )
    return out


def make_knee_point(frontier: list[dict], lam: float = 10.0) -> dict | None:
    """Return argmax_p (accuracy_match - lam * cost) over the frontier (excluding origin)."""
    if len(frontier) <= 1:
        return None
    candidates = frontier[1:]  # skip the (0, 0) origin
    best = max(candidates, key=lambda p: p["accuracy_match"] - lam * p["cost"])
    return {
        "cost":           best["cost"],
        "accuracy_match": best["accuracy_match"],
        "n_rules":        len(best["rules_admitted"]),
        "rules":          list(best["rules_admitted"]),
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def run_selection_pareto(
    rules_dir: str,
    eval_merge_path: Path,
    eval_individual_dir: Path,
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    labels: dict,
    cost_profile: dict[str, dict],
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    use_proxy: bool = False,
    tau_safety_floor: float = 0.0,
    knee_lambda: float = 10.0,
    query_thresholds: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 1.0),
) -> dict[str, Any]:
    """Run the Pareto-frontier selection pipeline.

    Reuses the upstream artefacts produced by the static and auto-tighten
    pipelines (cost_profile, eval_merge, eval_individual). The only new
    behaviour is the sort key (cov / cost descending) and the frontier
    recording.

    Args mirror `select_rules.run_selection`. `tau_safety_floor` defaults to 0.0
    (no τ filtering); set to e.g. 0.10 for a cheap safety filter on `S`.

    Returns a dict with frontier, query_table, knee_point, selected_rules,
    covered_docs, baseline_accuracy, selector_accuracy, llm_calls, token_usage.
    """
    eval_data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    baseline_accuracy = eval_data.get("accuracy", 0.0)
    target_docs = load_target_docs(eval_merge_path)
    n_docs = len(documents)

    print(f"  D* = {len(target_docs)} / {n_docs} docs  baseline_accuracy={baseline_accuracy:.2f}")

    if not target_docs:
        return _empty_pareto_result(
            question, question_slug, baseline_accuracy, tau_safety_floor,
        )

    # ── Phase 0.5 — coverage source for every rule in the pool ───────────────
    rule_pool = list(cost_profile.keys())
    cov_map = _build_cov_map(rule_pool, eval_individual_dir)

    # ── Phase 2 — cost-effectiveness sort + greedy cover ─────────────────────
    rules_sorted = _sort_by_cost_effectiveness(rule_pool, cov_map, cost_profile)
    print(f"  Sorted by cov/cost descending; top-3:")
    for r in rules_sorted[:3]:
        c = cost_profile[r].get("avg_cost_ratio", 0.0)
        v = cov_map.get(r, 0.0)
        print(f"    {r:<55}  cov={v:.2f}  cost={c:.6f}  ratio={v/(c+1e-9):.1f}")

    selected, per_rule_gained, qa_calls, judge_calls, in_tok, out_tok = _greedy_cover(
        rules_sorted=rules_sorted,
        target_docs=target_docs,
        documents=documents,
        question_slug=question_slug,
        question=question,
        rules_dir=rules_dir,
        labels=labels,
        model_name=model_name,
        output_dir=output_dir,
        use_proxy=use_proxy,
    )

    total_qa_calls = qa_calls
    total_judge_calls = judge_calls
    total_input_tokens = in_tok
    total_output_tokens = out_tok

    # ── Phase 3 (optional) — τ safety floor ──────────────────────────────────
    banned_global: set[str] = set()
    if tau_safety_floor > 0.0:
        for _ in range(10):
            filtered_S, docs_to_recover = filter_by_tau(
                selected_rules=selected,
                per_rule_gained=per_rule_gained,
                eval_individual_dir=eval_individual_dir,
                tau=tau_safety_floor,
            )
            newly_banned = set(selected) - set(filtered_S)
            banned_global |= newly_banned

            if not newly_banned:
                break

            for r in newly_banned:
                per_rule_gained.pop(r, None)
            selected = filtered_S

            if docs_to_recover:
                remaining = [
                    r for r in rules_sorted
                    if r not in set(selected) and r not in banned_global
                ]
                S_extra, gained_extra, qa, jc, in_tok2, out_tok2 = _greedy_cover(
                    rules_sorted=remaining,
                    target_docs=docs_to_recover,
                    documents=documents,
                    question_slug=question_slug,
                    question=question,
                    rules_dir=rules_dir,
                    labels=labels,
                    model_name=model_name,
                    output_dir=output_dir,
                    use_proxy=use_proxy,
                    initial_S=selected,
                )
                total_qa_calls     += qa
                total_judge_calls  += jc
                total_input_tokens  += in_tok2
                total_output_tokens += out_tok2
                selected += S_extra
                per_rule_gained.update(gained_extra)

    # ── Phase 4 — frontier reconstruction and summaries ──────────────────────
    frontier    = _build_frontier(selected, per_rule_gained, cov_map, cost_profile, target_docs)
    query_table = make_query_table(frontier, thresholds=query_thresholds)
    knee_point  = make_knee_point(frontier, lam=knee_lambda)

    covered = set().union(*per_rule_gained.values()) if per_rule_gained else set()
    uncovered = target_docs - covered
    selected_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in selected)

    return {
        "question":                    question,
        "question_slug":               question_slug,
        "mode":                        "pareto",
        "baseline_accuracy":           round(baseline_accuracy, 4),
        "selected_rules":              selected,
        "selected_avg_cost_ratio_sum": round(selected_cost, 6),
        "covered_docs":                sorted(covered),
        "uncovered_docs":              sorted(uncovered),
        "selector_accuracy":           round(len(covered) / n_docs, 4) if n_docs else 0.0,
        "tau_safety_floor":            tau_safety_floor,
        "banned_rules":                sorted(banned_global),
        "frontier":                    frontier,
        "query_table":                 query_table,
        "knee_point":                  knee_point,
        "knee_lambda":                 knee_lambda,
        "llm_calls": {
            "phase_2_incremental": total_qa_calls,
            "phase_2_judge":       total_judge_calls,
        },
        "token_usage": {
            "total_input_tokens":  total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_pareto_result(
    question: str,
    question_slug: str,
    baseline_accuracy: float,
    tau_safety_floor: float,
) -> dict[str, Any]:
    return {
        "question":                    question,
        "question_slug":               question_slug,
        "mode":                        "pareto",
        "baseline_accuracy":           round(baseline_accuracy, 4),
        "selected_rules":              [],
        "selected_avg_cost_ratio_sum": 0.0,
        "covered_docs":                [],
        "uncovered_docs":              [],
        "selector_accuracy":           0.0,
        "tau_safety_floor":            tau_safety_floor,
        "banned_rules":                [],
        "frontier":                    [{
            "cost":           0.0,
            "accuracy_match": 0.0,
            "rules_admitted": [],
            "added_rule":     None,
            "rule_cov":       None,
            "rule_avg_cost":  None,
        }],
        "query_table":                 {},
        "knee_point":                  None,
        "knee_lambda":                 10.0,
        "llm_calls":   {"phase_2_incremental": 0, "phase_2_judge": 0},
        "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0},
    }
