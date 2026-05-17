"""Pareto-frontier rule selection — proxy-only variant (zero LLM in selection).

Replaces the gpt54mini judge inside `_greedy_cover` with the substring proxy:
a rule is admitted iff its merged retrieved text contains the ground-truth
string on at least one new doc. The QA + judge LLM calls are skipped entirely
during selection. Phase 2 (sampled / unsampled eval with gpt54) is unchanged.

This module is additive — it does not modify `select_rules.py`,
`select_rules_auto_tighten.py`, or `select_rules_pareto.py`.

Reuses from select_rules_pareto.py:
  - _sort_by_cost_effectiveness
  - _build_frontier
  - make_query_table
  - make_knee_point
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rule_refinement.eval_judge          import proxy_judge
from rule_refinement.baseline_targets    import load_target_docs
from rule_refinement.coverage_check      import load_or_compute_coverage, filter_by_tau
from rule_refinement.select_rules_pareto import (
    _sort_by_cost_effectiveness,
    _build_frontier,
    make_query_table,
    make_knee_point,
)

_DEFAULT_OUTPUT_DIR = (
    "results/financebench_single_cluster/llm/gpt54/one_shot/selector_run_pareto_proxy"
)


# ── Tokenisation (no LLM) ─────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


# ── Rule loading (mirrors rule_apply_merge._load_rule_fn) ─────────────────────

def _load_rule_fn(rule_file: Path):
    spec = importlib.util.spec_from_file_location("_rule_mod", str(rule_file))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return next(v for k, v in vars(mod).items() if k.startswith("rule_") and callable(v))


# ── Retrieval-only merge (mirrors rule_apply_merge but no LLM) ────────────────

def _retrieve_merge(
    document:  dict,
    rule_names: list[str],
    rule_folder: Path,
) -> tuple[str, int]:
    """Apply rules, dedupe spans, sort in reading order, concat. Returns
    (retrieved_text, retrieved_token_count). No QA, no judge."""
    texts: list[dict] = document.get("texts", [])
    text_positions: dict[int, int] = {id(s): i for i, s in enumerate(texts)}

    all_retrieved: list[dict] = []
    for rn in rule_names:
        rule_file = rule_folder / f"{rn}.py"
        if not rule_file.exists():
            continue
        try:
            fn = _load_rule_fn(rule_file)
            spans = fn(document)
        except Exception as exc:
            warnings.warn(f"Rule {rn} error: {exc}")
            continue
        if spans:
            all_retrieved.extend(spans)

    # Dedupe by texts-index
    seen: set[int] = set()
    union: list[dict] = []
    for span in all_retrieved:
        idx = text_positions.get(id(span))
        if idx is None:
            try:
                idx = texts.index(span)
            except ValueError:
                idx = None
        if idx is None or idx not in seen:
            if idx is not None:
                seen.add(idx)
            union.append(span)

    def _sort_key(span: dict) -> tuple:
        page = span.get("page_no", 0)
        structure = span.get("structure") or {}
        level_index = structure.get("level_index")
        if level_index is None:
            level_index = text_positions.get(id(span), 0)
        return (page, level_index)

    sorted_spans = sorted(union, key=_sort_key)
    retrieved_text = "\n\n".join(s["text"] for s in sorted_spans) if sorted_spans else ""
    return retrieved_text, _count_tokens(retrieved_text)


# ── Greedy cover with proxy judge (no LLM) ────────────────────────────────────

def _greedy_cover_proxy(
    rules_sorted: list[str],
    target_docs:  set[str],
    documents:    dict[str, dict],
    question:     str,
    rule_folder:  Path,
    labels:       dict,
    initial_S:    list[str] | None = None,
) -> tuple[list[str], dict[str, set[str]]]:
    """Greedy cover using the substring proxy as the admission criterion.

    A rule r is admitted iff retrieved_text(working_S + [r]) contains the GT
    string for at least one still-uncovered doc. No LLM calls.

    Returns (newly_selected, per_rule_gained).
    """
    working_S:     list[str]            = list(initial_S or [])
    newly_selected: list[str]           = []
    U:              set[str]            = set(target_docs)
    per_rule_gained: dict[str, set[str]] = {}

    for r in rules_sorted:
        if not U:
            break

        gained: set[str] = set()
        for d in list(U):
            gt = labels.get(d + ".pdf", {}).get(question)
            if gt is None:
                continue
            doc = documents.get(d)
            if doc is None:
                continue
            try:
                retrieved_text, _ = _retrieve_merge(doc, working_S + [r], rule_folder)
            except Exception as exc:
                warnings.warn(f"retrieve_merge error on {d}: {exc}")
                continue
            if proxy_judge(gt, retrieved_text):
                gained.add(d)

        if gained:
            working_S.append(r)
            newly_selected.append(r)
            per_rule_gained[r] = gained
            U -= gained
            print(f"  + {r:<55}  gained={len(gained)}  remaining={len(U)}")

    return newly_selected, per_rule_gained


# ── Main entry point ─────────────────────────────────────────────────────────

def run_selection_pareto_proxy(
    rules_dir:           str,
    eval_merge_path:     Path,
    eval_individual_dir: Path,
    documents:           dict[str, dict],
    question_slug:       str,
    question:            str,
    labels:              dict,
    cost_profile:        dict[str, dict],
    output_dir:          str = _DEFAULT_OUTPUT_DIR,
    tau_safety_floor:    float = 0.0,
    knee_lambda:         float = 10.0,
    query_thresholds:    tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 1.0),
) -> dict[str, Any]:
    """Pareto selection where the in-loop judge is the substring proxy.

    No LLM calls during selection. Phase 2 (gpt54 sampled/unsampled eval) is
    handled by separate scripts. Output schema mirrors the original
    `run_selection_pareto` for downstream compatibility.
    """
    eval_data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    baseline_accuracy = eval_data.get("accuracy", 0.0)
    target_docs = load_target_docs(eval_merge_path)
    n_docs = len(documents)

    print(f"  D* = {len(target_docs)} / {n_docs} docs  baseline_accuracy={baseline_accuracy:.2f}")

    if not target_docs:
        return _empty(question, question_slug, baseline_accuracy, tau_safety_floor)

    rule_pool = list(cost_profile.keys())

    # cov(r) source for the cost-effectiveness sort
    cov_map = {
        r: load_or_compute_coverage(
            r,
            eval_individual_dir / f"{r}_eval.json",
            lambda: 0.0,
        )
        for r in rule_pool
    }

    rule_folder = Path(rules_dir) / question_slug

    rules_sorted = _sort_by_cost_effectiveness(rule_pool, cov_map, cost_profile)
    print(f"  Sorted by cov/cost descending; top-3:")
    for r in rules_sorted[:3]:
        c = cost_profile[r].get("avg_cost_ratio", 0.0)
        v = cov_map.get(r, 0.0)
        print(f"    {r:<55}  cov={v:.2f}  cost={c:.6f}  ratio={v/(c+1e-9):.1f}")

    # Phase 2 — proxy-only greedy
    selected, per_rule_gained = _greedy_cover_proxy(
        rules_sorted=rules_sorted,
        target_docs=target_docs,
        documents=documents,
        question=question,
        rule_folder=rule_folder,
        labels=labels,
    )

    # Phase 3 (optional safety floor) — mirror select_rules.py's ban-and-resume
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
                S_extra, gained_extra = _greedy_cover_proxy(
                    rules_sorted=remaining,
                    target_docs=docs_to_recover,
                    documents=documents,
                    question=question,
                    rule_folder=rule_folder,
                    labels=labels,
                    initial_S=selected,
                )
                selected += S_extra
                per_rule_gained.update(gained_extra)

    # Phase 4 — frontier + summaries
    frontier    = _build_frontier(selected, per_rule_gained, cov_map, cost_profile, target_docs)
    query_table = make_query_table(frontier, thresholds=query_thresholds)
    knee_point  = make_knee_point(frontier, lam=knee_lambda)

    covered = set().union(*per_rule_gained.values()) if per_rule_gained else set()
    uncovered = target_docs - covered
    selected_cost = sum(cost_profile.get(r, {}).get("avg_cost_ratio", 0.0) for r in selected)

    return {
        "question":                    question,
        "question_slug":               question_slug,
        "mode":                        "pareto_proxy",
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
            "phase_2_incremental":  0,
            "phase_2_judge":        0,
        },
        "token_usage": {
            "total_input_tokens":  0,
            "total_output_tokens": 0,
        },
    }


def _empty(question, slug, base, tau):
    return {
        "question":                    question,
        "question_slug":               slug,
        "mode":                        "pareto_proxy",
        "baseline_accuracy":           round(base, 4),
        "selected_rules":              [],
        "selected_avg_cost_ratio_sum": 0.0,
        "covered_docs":                [],
        "uncovered_docs":              [],
        "selector_accuracy":           0.0,
        "tau_safety_floor":            tau,
        "banned_rules":                [],
        "frontier":                    [{
            "cost": 0.0, "accuracy_match": 0.0, "rules_admitted": [],
            "added_rule": None, "rule_cov": None, "rule_avg_cost": None,
        }],
        "query_table":                 {},
        "knee_point":                  None,
        "knee_lambda":                 10.0,
        "llm_calls":   {"phase_2_incremental": 0, "phase_2_judge": 0},
        "token_usage": {"total_input_tokens": 0, "total_output_tokens": 0},
    }
