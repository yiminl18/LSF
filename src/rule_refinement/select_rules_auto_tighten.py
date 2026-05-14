"""Phase 2 + Phase 3 + Phase 4 (auto-tighten): maximum feasible tau.

Builds on the static-tau pipeline (select_rules.py) and adds a tightening
loop that iteratively raises tau by banning the lowest-coverage rule in S
and replacing it with higher-coverage alternatives, until no further
tightening is possible without dropping a doc from D*.

max_iters=10 is the tight bound: with m=10 sampled docs cov(r) takes values
in {0.0, 0.1, ..., 1.0}, so at most 10 distinct levels can be crossed.
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

from rule_apply_merge import rule_apply_merge
from rule_refinement.select_rules import _greedy_cover
from rule_refinement.eval_judge import judge
from rule_refinement.baseline_targets import load_target_docs
from rule_refinement.coverage_check import filter_by_tau, load_or_compute_coverage

_DEFAULT_OUTPUT_DIR = (
    "results/financebench_single_cluster/llm/gpt54/refine_dynamic_generality/selector_run_auto"
)


def _backward_prune(
    selected: list[str],
    per_rule_gained: dict[str, set[str]],
    target_docs: set[str],
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    rules_dir: str,
    labels: dict,
    model_name: str,
    output_dir: str,
) -> tuple[list[str], dict[str, set[str]], int, int, int, int]:
    """Try removing each rule from selected (most expensive first).

    A rule is dropped if the remaining set still correctly answers every doc in
    target_docs. Iterates from the last-added (most expensive) rule to the first.

    Returns (pruned_rules, updated_per_rule_gained, qa_calls, judge_calls,
             total_input_tokens, total_output_tokens).
    """
    qa_calls = 0
    judge_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    current = list(selected)

    i = len(current) - 1
    while i >= 0:
        r = current[i]
        candidate = [x for x in current if x != r]

        if not candidate:
            i -= 1
            continue

        covered: set[str] = set()
        failed = False
        for d in target_docs:
            gt = labels.get(d + ".pdf", {}).get(question)
            try:
                res = rule_apply_merge(
                    document=documents[d],
                    rule_names=candidate,
                    question_slug=question_slug,
                    question=question,
                    model_name=model_name,
                    rules_dir=rules_dir,
                    output_dir=output_dir,
                )
            except Exception as exc:
                print(f"    PRUNE SKIP {d} (error: {exc})")
                failed = True
                break
            qa_calls += 1
            total_input_tokens += res.get("input_tokens", 0)
            total_output_tokens += res.get("output_tokens", 0)

            try:
                correct, j_in, j_out = judge(question, gt, res["predicted_answer"], model_name=model_name)
                if correct:
                    covered.add(d)
                judge_calls += 1
                total_input_tokens += j_in
                total_output_tokens += j_out
            except Exception as exc:
                print(f"    PRUNE SKIP judge {d} (error: {exc})")
                failed = True
                break

        if not failed and target_docs <= covered:
            current = candidate
            removed_docs = per_rule_gained.pop(r, set())
            # Redistribute docs that were uniquely credited to removed rule.
            still_credited = set().union(*per_rule_gained.values()) if per_rule_gained else set()
            orphaned = removed_docs - still_credited
            if orphaned and current:
                per_rule_gained.setdefault(current[-1], set()).update(orphaned)
            print(f"  PRUNE removed {r:<55}  D* still covered")
        else:
            print(f"  PRUNE kept    {r:<55}  losing {len(target_docs - covered)} doc(s)")

        i -= 1

    return current, per_rule_gained, qa_calls, judge_calls, total_input_tokens, total_output_tokens


def run_selection_auto_tighten(
    rules_dir: str,
    eval_merge_path: Path,
    eval_individual_dir: Path,
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    labels: dict,
    cost_profile: dict[str, dict],
    tau_floor: float = 0.20,
    epsilon: float = 1e-6,
    max_iters: int = 10,
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    use_proxy: bool = False,
    backtracking: bool = True,
) -> dict[str, Any]:
    """Run Phase 2 + Phase 3 (static tau_floor) + optional backward pruning + Phase 4 (auto-tighten).

    Phase 3.5 (backtracking=True): after the greedy cover converges, try removing
    each rule from most-expensive to cheapest. A rule is dropped if the remaining
    set still covers all of D*. Cost: at most |S| * |D*| extra LLM calls.

    Phase 4 iteratively bans the lowest-coverage rule in S and tries to replace
    it with higher-coverage alternatives. Each accepted step raises tau_best while
    preserving full coverage of D*. Stops when no further tightening is feasible.

    Returns dict with keys:
        question, question_slug, mode, tau_floor, tau_best,
        selected_rules, selected_avg_cost_ratio_sum,
        covered_docs, uncovered_docs,
        baseline_accuracy, selector_accuracy,
        tightening_history,
        llm_calls: {phase_2_incremental, phase_3_coverage, phase_35_backtracking, phase_4_auto_tighten}
    """
    eval_data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    baseline_accuracy = eval_data.get("accuracy", 0.0)
    target_docs = load_target_docs(eval_merge_path)
    n_docs = len(documents)

    print(f"  D* = {len(target_docs)} / {n_docs} docs  baseline_accuracy={baseline_accuracy:.2f}")

    if not target_docs:
        return {
            "question": question,
            "question_slug": question_slug,
            "mode": "auto_tighten",
            "tau_floor": tau_floor,
            "tau_best": tau_floor,
            "selected_rules": [],
            "selected_avg_cost_ratio_sum": 0.0,
            "covered_docs": [],
            "uncovered_docs": [],
            "baseline_accuracy": round(baseline_accuracy, 4),
            "selector_accuracy": 0.0,
            "tightening_history": [],
            "llm_calls": {"phase_2_incremental": 0, "phase_3_coverage": 0, "phase_35_backtracking": 0, "phase_4_auto_tighten": 0},
            "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0},
        }

    rules_sorted = sorted(
        cost_profile.keys(),
        key=lambda r: cost_profile[r].get("avg_cost_ratio", float("inf")),
    )

    # Build cov_map for the full pool — used to gate admissibility in Phase 4.
    cov_map: dict[str, float] = {}
    for r in rules_sorted:
        eval_path = eval_individual_dir / f"{r}_eval.json"
        cov_map[r] = load_or_compute_coverage(r, eval_path, lambda: 0.0)

    total_qa_calls = 0
    total_judge_calls = 0
    phase4_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # ── Phase 2: initial greedy cover ────────────────────────────────────────
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
        use_proxy=use_proxy,
    )
    total_qa_calls += qa
    total_judge_calls += jc
    total_input_tokens += in_tok
    total_output_tokens += out_tok

    # ── Phase 3: tau_floor check with ban-and-resume ──────────────────────────
    banned_global: set[str] = set()

    for _ in range(10):
        filtered_S, docs_to_recover = filter_by_tau(
            selected_rules=S,
            per_rule_gained=per_rule_gained,
            eval_individual_dir=eval_individual_dir,
            tau=tau_floor,
        )
        newly_banned = set(S) - set(filtered_S)
        banned_global |= newly_banned

        if not newly_banned:
            break

        for r in newly_banned:
            per_rule_gained.pop(r, None)
        S = filtered_S

        if docs_to_recover:
            remaining = [
                r for r in rules_sorted
                if r not in set(S) and r not in banned_global
            ]
            S_extra, gained_extra, qa, jc, in_tok, out_tok = _greedy_cover(
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
                initial_S=S,
            )
            total_qa_calls += qa
            total_judge_calls += jc
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            S += S_extra
            per_rule_gained.update(gained_extra)

    # ── Phase 3.5: backward pruning (optional) ───────────────────────────────
    backtrack_calls = 0
    backtrack_input_tokens = 0
    backtrack_output_tokens = 0
    if backtracking and S:
        print(f"\n  Phase 3.5 backtracking: |S|={len(S)}")
        S, per_rule_gained, qa, jc, in_tok, out_tok = _backward_prune(
            selected=S,
            per_rule_gained=per_rule_gained,
            target_docs=target_docs,
            documents=documents,
            question_slug=question_slug,
            question=question,
            rules_dir=rules_dir,
            labels=labels,
            model_name=model_name,
            output_dir=output_dir,
        )
        backtrack_calls += qa + jc
        backtrack_input_tokens += in_tok
        backtrack_output_tokens += out_tok
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        print(f"  Phase 3.5 done: |S|={len(S)}")

    # ── Phase 4: auto-tighten ─────────────────────────────────────────────────
    tau_best = min((cov_map.get(r, 0.0) for r in S), default=tau_floor)
    tightening_history: list[dict] = []
    phase4_input_tokens = 0
    phase4_output_tokens = 0

    print(f"\n  Phase 4 auto-tighten: |S|={len(S)}  tau_floor={tau_floor}  tau_best={tau_best:.3f}")

    for iteration in range(max_iters):
        if not S:
            break

        r_min = min(S, key=lambda r: cov_map.get(r, 0.0))
        cov_min = cov_map.get(r_min, 0.0)
        tau_try = cov_min + epsilon

        remaining_S = [r for r in S if r != r_min]
        remaining_covered = (
            set().union(*(per_rule_gained.get(r, set()) for r in remaining_S))
            if remaining_S else set()
        )
        docs_to_recover = per_rule_gained.get(r_min, set()) - remaining_covered

        admissible = [
            r for r in rules_sorted
            if r not in set(S) and r not in banned_global and cov_map.get(r, 0.0) >= tau_try
        ]

        print(
            f"  Iter {iteration + 1}: ban {r_min} (cov={cov_min:.2f})  "
            f"tau_try={tau_try:.4f}  recover={len(docs_to_recover)}  admissible={len(admissible)}"
        )

        if not docs_to_recover:
            # r_min's docs are already covered by remaining S — free drop.
            S = remaining_S
            per_rule_gained.pop(r_min, None)
            banned_global.add(r_min)
            tau_best = min((cov_map.get(r, 0.0) for r in S), default=tau_floor)
            tightening_history.append({
                "iteration": iteration + 1,
                "banned_rule": r_min,
                "cov_banned": round(cov_min, 4),
                "tau_try": round(tau_try, 6),
                "added_rules": [],
                "added_covs": {},
                "below_tau_try": [],
                "feasible_cover": True,
                "accepted": True,
            })
            print(f"    Free drop — tau_best now {tau_best:.3f}")
            continue

        if not admissible:
            tightening_history.append({
                "iteration": iteration + 1,
                "banned_rule": r_min,
                "cov_banned": round(cov_min, 4),
                "tau_try": round(tau_try, 6),
                "added_rules": [],
                "added_covs": {},
                "below_tau_try": [],
                "feasible_cover": False,
                "accepted": False,
            })
            print(f"    No admissible replacements — stop.")
            break

        S_extra, gained_extra, qa, jc, in_tok, out_tok = _greedy_cover(
            rules_sorted=admissible,
            target_docs=docs_to_recover,
            documents=documents,
            question_slug=question_slug,
            question=question,
            rules_dir=rules_dir,
            labels=labels,
            model_name=model_name,
            output_dir=output_dir,
            use_proxy=use_proxy,
            initial_S=remaining_S,
        )
        phase4_calls += qa + jc
        phase4_input_tokens += in_tok
        phase4_output_tokens += out_tok
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        candidate_gained = {r: per_rule_gained[r] for r in remaining_S}
        candidate_gained.update(gained_extra)
        candidate_covered = set().union(*candidate_gained.values()) if candidate_gained else set()
        feasible = target_docs <= candidate_covered

        below_tau_try = [r for r in S_extra if cov_map.get(r, 0.0) < tau_try]

        entry = {
            "iteration": iteration + 1,
            "banned_rule": r_min,
            "cov_banned": round(cov_min, 4),
            "tau_try": round(tau_try, 6),
            "added_rules": S_extra,
            "added_covs": {r: round(cov_map.get(r, 0.0), 4) for r in S_extra},
            "below_tau_try": below_tau_try,
            "feasible_cover": feasible,
            "accepted": feasible,
        }
        tightening_history.append(entry)

        if feasible:
            S = remaining_S + S_extra
            per_rule_gained = candidate_gained
            banned_global.add(r_min)
            tau_best = min((cov_map.get(r, 0.0) for r in S), default=tau_floor)
            print(f"    Accepted — tau_best now {tau_best:.3f}  |S|={len(S)}")
        else:
            print(
                f"    Rejected (covers {len(candidate_covered)}/{len(target_docs)} of D*) — stop."
            )
            break

    # ── Final accounting ──────────────────────────────────────────────────────
    covered = set().union(*per_rule_gained.values()) if per_rule_gained else set()
    uncovered = target_docs - covered
    selected_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in S)

    return {
        "question": question,
        "question_slug": question_slug,
        "mode": "auto_tighten",
        "tau_floor": tau_floor,
        "tau_best": round(tau_best, 4),
        "selected_rules": S,
        "selected_avg_cost_ratio_sum": round(selected_cost, 6),
        "covered_docs": sorted(covered),
        "uncovered_docs": sorted(uncovered),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "selector_accuracy": round(len(covered) / n_docs, 4) if n_docs else 0.0,
        "tightening_history": tightening_history,
        "llm_calls": {
            "phase_2_incremental": total_qa_calls,
            "phase_3_coverage": total_judge_calls,
            "phase_35_backtracking": backtrack_calls,
            "phase_4_auto_tighten": phase4_calls,
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "phase_2_3_input": total_input_tokens - backtrack_input_tokens - phase4_input_tokens,
            "phase_2_3_output": total_output_tokens - backtrack_output_tokens - phase4_output_tokens,
            "phase_35_input": backtrack_input_tokens,
            "phase_35_output": backtrack_output_tokens,
            "phase_4_input": phase4_input_tokens,
            "phase_4_output": phase4_output_tokens,
        },
    }
