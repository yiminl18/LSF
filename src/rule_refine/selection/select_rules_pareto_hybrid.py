"""Pareto-frontier rule selection — hybrid model variant.

Decouples the **coverage estimation** model from the **verification** model:

  - **Coverage** (used as the per-rule sort key in cost-effectiveness ordering):
    use **gpt-5.4-mini** judgements. The noisier-but-cheap signal acts as a
    regularizer for which specialist rules to even consider, while costing far
    less to precompute.

  - **In-loop admission** (does this candidate rule cover a previously-uncovered
    sampled doc?) **and final merge verification** (does the union of selected
    rules match base accuracy?): use **gpt-5.4**. The strong judge is what
    actually decides whether a rule earns its slot in `S`.

The intuition: gpt-5.4-mini is good enough at "is this passage roughly about
the right thing" (coverage) but unreliable for "is the answer correct"
(verification). Use it where it shines, and reserve the expensive model for
where correctness matters.

The implementation is additive — does not modify any earlier Pareto variant.
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

from rule_apply.merge                          import rule_apply_merge
from rule_refine.selection.eval_judge          import judge
from rule_refine.selection.baseline_targets    import load_target_docs
from rule_refine.selection.coverage_check      import load_or_compute_coverage
from rule_refine.selection.select_rules        import _greedy_cover
from rule_refine.selection.select_rules_pareto import (
    _sort_by_cost_effectiveness,
    _build_frontier,
    make_query_table,
    make_knee_point,
)

_DEFAULT_OUTPUT_DIR = (
    "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/selector_run_pareto_hybrid"
)


# Model split — the whole point of this variant.
_COVERAGE_MODEL = "gpt54mini"
_VERIFY_MODEL   = "gpt54"


def _build_cov_map_from_dir(rule_pool: list[str], eval_dir: Path) -> dict[str, float]:
    """Read per-rule coverage from a precomputed eval_individual-style dir.

    Mirrors `select_rules_pareto._build_cov_map` but accepts an arbitrary dir
    (so we can point at a gpt54mini-judged cache instead of the default
    gpt54-judged one).
    """
    cov: dict[str, float] = {}
    for r in rule_pool:
        eval_path = eval_dir / f"{r}_eval.json"
        cov[r] = load_or_compute_coverage(r, eval_path, fallback=lambda: 0.0)
    return cov


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
) -> tuple[int, int, int, int, int, int]:
    """Run rule_apply_merge + LLM judge on `doc_names`. Returns
    (n_correct, n, qa_calls, judge_calls, in_tokens, out_tokens).
    Same shape as `select_rules_pareto_v2._eval_merge_on_docs`.
    """
    n_correct = 0
    qa_calls  = 0
    judge_calls = 0
    in_tok    = 0
    out_tok   = 0

    for d in doc_names:
        if d not in documents:
            continue
        try:
            res = rule_apply_merge(
                document      = documents[d],
                rule_names    = rule_names,
                question_slug = question_slug,
                question      = question,
                model_name    = model_name,
                rules_dir     = rules_dir,
                output_dir    = output_dir,
            )
            qa_calls += 1
            gt = labels.get(d + ".pdf", {}).get(question)
            correct, j_in, j_out = judge(
                question, gt, res["predicted_answer"], model_name=model_name,
            )
            judge_calls += 1
            in_tok  += int(res.get("input_tokens", 0) or 0) + int(j_in or 0)
            out_tok += int(res.get("output_tokens", 0) or 0) + int(j_out or 0)
            if correct:
                n_correct += 1
        except Exception as exc:
            print(f"    merge-eval error on {d}: {exc}")

    return n_correct, len(doc_names), qa_calls, judge_calls, in_tok, out_tok


# ── Main entry point ─────────────────────────────────────────────────────────

def run_selection_pareto_hybrid(
    rules_dir: str,
    eval_merge_path: Path,
    eval_individual_dir: Path,                # gpt54-judged (kept for fallback)
    documents: dict[str, dict],
    question_slug: str,
    question: str,
    labels: dict,
    cost_profile: dict[str, dict],
    coverage_eval_individual_dir: Path | None = None,    # gpt54mini-judged cov cache
    coverage_model: str = _COVERAGE_MODEL,               # only used if cov cache missing
    verify_model:   str = _VERIFY_MODEL,                  # in-loop + final merge judge
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    use_proxy: bool = False,
    query_thresholds: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 1.0),
    knee_lambda: float = 10.0,
) -> dict[str, Any]:
    """Run the hybrid Pareto pipeline (mini-coverage, gpt54-verification).

    Args:
        coverage_eval_individual_dir: directory of `<rule>_eval.json` files
            whose `accuracy` field was computed under `coverage_model` (e.g.
            gpt54mini). If None or missing, falls back to `eval_individual_dir`.
        coverage_model: model assumed to have produced the coverage cache.
            Informational only (the cache is the source of truth).
        verify_model: model used by `_greedy_cover` for the in-loop admission
            check and by `_eval_merge_on_docs` for the final merge accuracy
            verification.

    Returns the standard selector dict (selected_rules, baseline_accuracy,
    selector_accuracy, llm_calls, token_usage, plus a `phases` block recording
    which model was used where).
    """
    eval_data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    baseline_accuracy = eval_data.get("accuracy", 0.0)
    target_docs = load_target_docs(eval_merge_path)
    n_docs = len(documents)

    print(f"  D* = {len(target_docs)} / {n_docs} docs  baseline_accuracy={baseline_accuracy:.2f}")
    print(f"  coverage_model={coverage_model}  verify_model={verify_model}")

    if not target_docs:
        return {
            "question":           question,
            "question_slug":      question_slug,
            "mode":               "pareto_hybrid",
            "selected_rules":     [],
            "selector_accuracy":  baseline_accuracy,
            "baseline_accuracy":  baseline_accuracy,
            "covered_docs":       [],
            "llm_calls":          {"phase_2_qa": 0, "phase_2_judge": 0, "phase_B_qa": 0, "phase_B_judge": 0},
            "token_usage":        {"input": 0, "output": 0},
            "phases":             {"coverage": coverage_model, "verify": verify_model},
        }

    # ── Phase 0 — pick the coverage source ──────────────────────────────────
    eff_cov_dir = (
        Path(coverage_eval_individual_dir)
        if coverage_eval_individual_dir is not None
        else Path(eval_individual_dir)
    )
    if not eff_cov_dir.exists():
        warnings.warn(
            f"coverage cache dir not found at {eff_cov_dir}; "
            f"falling back to {eval_individual_dir} (verify_model-judged)."
        )
        eff_cov_dir = Path(eval_individual_dir)

    rule_pool = list(cost_profile.keys())
    cov_map = _build_cov_map_from_dir(rule_pool, eff_cov_dir)

    # ── Phase 1 — sort by cov(mini) / cost descending ───────────────────────
    rules_sorted = _sort_by_cost_effectiveness(rule_pool, cov_map, cost_profile)
    print(f"  Sorted by cov_{coverage_model}/cost descending; top-3:")
    for r in rules_sorted[:3]:
        c = cost_profile[r].get("avg_cost_ratio", 0.0)
        v = cov_map.get(r, 0.0)
        print(f"    {r:<55}  cov={v:.2f}  cost={c:.6f}  ratio={v/(c+1e-9):.1f}")

    # ── Phase 2 — greedy cover; in-loop verify uses verify_model ────────────
    selected, per_rule_gained, qa_calls, judge_calls, in_tok, out_tok = _greedy_cover(
        rules_sorted   = rules_sorted,
        target_docs    = target_docs,
        documents      = documents,
        question_slug  = question_slug,
        question       = question,
        rules_dir      = rules_dir,
        labels         = labels,
        model_name     = verify_model,
        output_dir     = output_dir,
        use_proxy      = use_proxy,
    )

    print(f"  Phase 2 selected {len(selected)} rules; "
          f"qa={qa_calls}  judge={judge_calls}  in={in_tok}  out={out_tok}")

    # ── Phase B — accuracy floor (verify_model on the union) ─────────────────
    # If the selected subset doesn't match baseline_accuracy on D*, admit
    # next-best by cov_mini/cost order until it does, mirroring p_v2 Phase B.
    target_list = sorted(target_docs)
    n_correct, n_total, qa_b, jud_b, in_tok_b, out_tok_b = _eval_merge_on_docs(
        rule_names    = selected,
        documents     = documents,
        doc_names     = target_list,
        question      = question,
        question_slug = question_slug,
        rules_dir     = rules_dir,
        labels        = labels,
        model_name    = verify_model,
        output_dir    = output_dir,
    )
    merge_acc = (n_correct / n_total) if n_total else 0.0
    print(f"  Phase B initial merge_acc on D*: {merge_acc:.3f}  (base={baseline_accuracy:.3f})")

    qa_calls_b    = qa_b
    judge_calls_b = jud_b
    in_tok_b_total  = in_tok_b
    out_tok_b_total = out_tok_b

    max_extra_rules = 20
    extras_added = 0
    while merge_acc + 1e-9 < baseline_accuracy and extras_added < max_extra_rules:
        # Pick the next rule from the cost-effectiveness ordering not yet in S
        next_rule = next((r for r in rules_sorted if r not in selected), None)
        if next_rule is None:
            break
        selected.append(next_rule)
        extras_added += 1
        n_correct, n_total, qa_b, jud_b, in_tok_b, out_tok_b = _eval_merge_on_docs(
            rule_names    = selected,
            documents     = documents,
            doc_names     = target_list,
            question      = question,
            question_slug = question_slug,
            rules_dir     = rules_dir,
            labels        = labels,
            model_name    = verify_model,
            output_dir    = output_dir,
        )
        merge_acc = (n_correct / n_total) if n_total else 0.0
        qa_calls_b    += qa_b
        judge_calls_b += jud_b
        in_tok_b_total  += in_tok_b
        out_tok_b_total += out_tok_b
        print(f"  Phase B extra +{next_rule!r} → merge_acc={merge_acc:.3f}  (extras_added={extras_added})")

    # ── Phase C — backward prune (drop redundant rules) ─────────────────────
    pruned = list(selected)
    qa_calls_c = judge_calls_c = 0
    in_tok_c = out_tok_c = 0
    for r in reversed(list(pruned)):
        if len(pruned) <= 1:
            break
        trial = [x for x in pruned if x != r]
        n_correct, n_total, qa_c, jud_c, in_tok_inc, out_tok_inc = _eval_merge_on_docs(
            rule_names    = trial,
            documents     = documents,
            doc_names     = target_list,
            question      = question,
            question_slug = question_slug,
            rules_dir     = rules_dir,
            labels        = labels,
            model_name    = verify_model,
            output_dir    = output_dir,
        )
        qa_calls_c   += qa_c
        judge_calls_c += jud_c
        in_tok_c  += in_tok_inc
        out_tok_c += out_tok_inc
        acc = (n_correct / n_total) if n_total else 0.0
        if acc + 1e-9 >= baseline_accuracy:
            pruned = trial
            print(f"  Phase C drop {r!r}  acc={acc:.3f}")

    selected = pruned

    covered_docs = sorted({d for s in selected for d in per_rule_gained.get(s, set())})
    frontier = _build_frontier(selected, per_rule_gained, cost_profile)
    query_table = make_query_table(frontier, query_thresholds)
    knee = make_knee_point(frontier, lam=knee_lambda)

    total_qa    = qa_calls + qa_calls_b + qa_calls_c
    total_judge = judge_calls + judge_calls_b + judge_calls_c
    total_in    = in_tok + in_tok_b_total + in_tok_c
    total_out   = out_tok + out_tok_b_total + out_tok_c

    final_acc = (n_correct / n_total) if n_total else 0.0

    return {
        "question":           question,
        "question_slug":      question_slug,
        "mode":               "pareto_hybrid",
        "selected_rules":     selected,
        "covered_docs":       covered_docs,
        "baseline_accuracy":  baseline_accuracy,
        "selector_accuracy":  final_acc,
        "frontier":           frontier,
        "query_table":        query_table,
        "knee_point":         knee,
        "phases": {
            "coverage": coverage_model,
            "verify":   verify_model,
            "Phase_B_extras_added": extras_added,
            "Phase_C_kept":         len(selected),
        },
        "llm_calls": {
            "phase_2_qa":    qa_calls,
            "phase_2_judge": judge_calls,
            "phase_B_qa":    qa_calls_b,
            "phase_B_judge": judge_calls_b,
            "phase_C_qa":    qa_calls_c,
            "phase_C_judge": judge_calls_c,
        },
        "token_usage": {"input": total_in, "output": total_out},
    }
