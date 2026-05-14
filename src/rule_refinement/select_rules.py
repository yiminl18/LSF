"""Phase 2 + Phase 3: cost-sorted incremental cover with tau filtering.

Phase 2 — greedy cover:
    Iterate rules in ascending avg_cost_ratio order.  For each candidate rule r,
    test S ∪ {r} on every still-uncovered doc.  Admit r iff it covers at least
    one new doc.  Use a substring proxy to skip the LLM judge when the ground-
    truth string is absent from the retrieved text.

Phase 3 — tau check with ban-and-resume:
    For every r ∈ S check cov(r) ≥ tau.  Ban rules that fail; re-run Phase 2
    on the docs they were responsible for (using the remaining rule pool as
    candidates).  Repeat until stable (≤ 10 rounds).
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
from rule_refinement.eval_judge import judge, proxy_judge
from rule_refinement.baseline_targets import load_target_docs
from rule_refinement.coverage_check import filter_by_tau

_DEFAULT_OUTPUT_DIR = (
    "results/financebench_single_cluster/llm/gpt54/one_shot/selector_run"
)


def _greedy_cover(
    rules_sorted: list[str],
    target_docs: set[str],
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    rules_dir: str,
    labels: dict,
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    use_proxy: bool = False,
    initial_S: list[str] | None = None,
) -> tuple[list[str], dict[str, set[str]], int, int, int, int]:
    """One greedy cover pass over rules_sorted.

    initial_S is the set of already-committed rules used as merge context; they
    are NOT returned in the output — only the newly selected rules are.

    Returns:
        newly_selected, per_rule_gained, qa_calls, judge_calls,
        total_input_tokens, total_output_tokens.
    """
    context_S: list[str] = list(initial_S or [])
    working_S: list[str] = list(context_S)
    newly_selected: list[str] = []
    U: set[str] = set(target_docs)
    per_rule_gained: dict[str, set[str]] = {}
    qa_calls = 0
    judge_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for r in rules_sorted:
        if not U:
            break

        gained: set[str] = set()

        for d in list(U):
            gt = labels.get(d + ".pdf", {}).get(question)

            try:
                res = rule_apply_merge(
                    document=documents[d],
                    rule_names=working_S + [r],
                    question_slug=question_slug,
                    question=question,
                    model_name=model_name,
                    rules_dir=rules_dir,
                    output_dir=output_dir,
                )
            except Exception as exc:
                print(f"    SKIP {d} (rule_apply_merge error: {exc})")
                continue
            qa_calls += 1
            total_input_tokens += res.get("input_tokens", 0)
            total_output_tokens += res.get("output_tokens", 0)

            retrieved_text = res.get("retrieved_text", "")

            # Skip expensive LLM judge when ground truth is absent from retrieved text.
            if use_proxy and not proxy_judge(gt, retrieved_text):
                continue

            try:
                correct, j_in, j_out = judge(question, gt, res["predicted_answer"], model_name=model_name)
                if correct:
                    gained.add(d)
                judge_calls += 1
                total_input_tokens += j_in
                total_output_tokens += j_out
            except Exception as exc:
                print(f"    SKIP judge {d} (error: {exc})")

        if gained:
            working_S.append(r)
            newly_selected.append(r)
            per_rule_gained[r] = gained
            U -= gained
            print(f"  + {r:<55}  gained={len(gained)}  remaining={len(U)}")

    return newly_selected, per_rule_gained, qa_calls, judge_calls, total_input_tokens, total_output_tokens


def run_selection(
    rules_dir: str,
    eval_merge_path: Path,
    eval_individual_dir: Path,
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    labels: dict,
    cost_profile: dict[str, dict],
    tau: float = 0.20,
    model_name: str = "gpt54",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    use_proxy: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 2 + Phase 3 selection algorithm.

    Args:
        rules_dir: base directory for rules (e.g. rules/.../one_shot);
                   rule files are at rules_dir/question_slug/rule_name.py.
        eval_merge_path: path to eval_merge/<slug>_sampled.json (Phase 1 input).
        eval_individual_dir: directory holding <rule>_eval.json files (Phase 3 input).
        documents: {doc_name: loaded doc dict}.
        question_slug: folder name under rules_dir (includes _llm suffix).
        question: natural-language question string.
        labels: ground-truth dict from sample_doc_labels.json.
        cost_profile: output of cost_profile.load_or_compute_cost_profile().
        tau: minimum per-rule coverage threshold.
        output_dir: where selector merge outputs are written (isolated from the
                    main rule_run_merge directory).
        use_proxy: whether to apply the substring proxy pre-filter.

    Returns dict with keys:
        question, question_slug, tau, selected_rules,
        selected_avg_cost_ratio_sum, covered_docs, uncovered_docs,
        baseline_accuracy, selector_accuracy,
        llm_calls: {phase_2_incremental, phase_3_coverage}
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
            "tau": tau,
            "selected_rules": [],
            "selected_avg_cost_ratio_sum": 0.0,
            "covered_docs": [],
            "uncovered_docs": [],
            "baseline_accuracy": round(baseline_accuracy, 4),
            "selector_accuracy": 0.0,
            "llm_calls": {"phase_2_incremental": 0, "phase_3_coverage": 0},
            "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0},
        }

    # Phase 0 output: rules sorted cheapest-first
    rules_sorted = sorted(
        cost_profile.keys(),
        key=lambda r: cost_profile[r].get("avg_cost_ratio", float("inf")),
    )

    total_qa_calls = 0
    total_judge_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Phase 2: initial greedy cover
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

    # Phase 3: tau check with ban-and-resume (up to 10 rounds)
    banned_global: set[str] = set()

    for _ in range(10):
        filtered_S, docs_to_recover = filter_by_tau(
            selected_rules=S,
            per_rule_gained=per_rule_gained,
            eval_individual_dir=eval_individual_dir,
            tau=tau,
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

    covered = set().union(*per_rule_gained.values()) if per_rule_gained else set()
    uncovered = target_docs - covered
    selected_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in S)

    return {
        "question": question,
        "question_slug": question_slug,
        "tau": tau,
        "selected_rules": S,
        "selected_avg_cost_ratio_sum": round(selected_cost, 6),
        "covered_docs": sorted(covered),
        "uncovered_docs": sorted(uncovered),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "selector_accuracy": round(len(covered) / n_docs, 4) if n_docs else 0.0,
        "llm_calls": {
            "phase_2_incremental": total_qa_calls,
            "phase_3_coverage": total_judge_calls,
        },
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
    }
