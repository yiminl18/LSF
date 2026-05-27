"""Main entry point for rule_refine_v2.

Orchestrates the seven stages:
  A. Per-rule profiling (LLM-free)
  B. Build D* target docs
  C. Composite utility
  D. Greedy set-cover
  E. Merge accuracy verification (LLM)
  F. k-fold stability filter (LLM via re-verification)
  G. Optional budget pruning (LLM)

See docs/rule_refine_v2.md.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .profile      import profile_rules
from .target_docs  import build_target_set, filter_useful_rules
from .utility      import rank_by_utility
from .set_cover    import greedy_cover
from .merge_verify import verify_and_extend
from .stability    import k_fold_selection, stable_rule_set


def rule_refine_v2(
    rule_names: list[str],
    question: str,
    question_slug: str,
    documents: list[dict],
    ground_truth: dict,
    rules_dir: str = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot",
    output_dir: str = "rules/financebench/lsf/single_cluster/llm/gpt54/refine_v2",
    model_name: str = "gpt54mini",
    target_accuracy: float | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-4,
    k_folds: int = 5,
    fold_frac: float = 0.7,
    min_folds: int | None = None,
    rule_budget: int | None = None,
    fuzzy_numeric: bool = True,
    profile_cache_dir: str | None = None,
    use_stability: bool = True,
    seed: int | None = 42,
    max_extra_rules: int = 5,
) -> dict:
    """Run the v2 refinement pipeline and write artifacts.

    target_accuracy default: if None, set to the proxy ceiling |D*| / |docs|.
    """
    model_mod = importlib.import_module(f"models.{model_name}")

    rule_folder = Path(rules_dir) / f"{question_slug}_llm"
    if not rule_folder.exists():
        rule_folder = Path(rules_dir) / question_slug
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.time()

    # ── Stage A: profile (LLM-free) ───────────────────────────────────────────
    cache_path = None
    if profile_cache_dir is not None:
        cache_path = Path(profile_cache_dir) / f"{question_slug}_profile.json"
    print(f"  [A] profiling {len(rule_names)} rules on {len(documents)} docs (no LLM)...", flush=True)
    t0 = time.time()
    profile = profile_rules(
        rule_names, documents, ground_truth, rule_folder,
        fuzzy_numeric=fuzzy_numeric, cache_path=cache_path,
    )
    print(f"  [A] done in {time.time()-t0:.1f}s; profiled {len(profile)} rules", flush=True)

    # ── Stage B: build D* ─────────────────────────────────────────────────────
    D_star = build_target_set(profile)
    all_doc_names = {
        (d.get("doc_name", d.get("origin", {}).get("filename", "unknown"))).replace(".pdf", "")
        for d in documents
    }
    print(f"  [B] D* = {len(D_star)}/{len(all_doc_names)} docs", flush=True)
    if not D_star:
        print("  [B] D* is empty (no rule retrieves the GT string). Aborting.", flush=True)
        return _empty_result(question, question_slug, timestamp, model_name, len(documents),
                             list(profile.keys()), out_dir, t_start)

    useful = filter_useful_rules(profile, D_star)
    print(f"  [B] useful rules (∩ D* ≠ ∅): {len(useful)}/{len(profile)}", flush=True)

    # Set target_accuracy from proxy ceiling unless user pinned it
    proxy_ceiling = len(D_star) / max(1, len(all_doc_names))
    if target_accuracy is None:
        target_accuracy = proxy_ceiling
    print(f"  [B] target_accuracy = {target_accuracy:.2f} (proxy ceiling = {proxy_ceiling:.2f})", flush=True)

    # ── Stage C: rank by utility ─────────────────────────────────────────────
    sub_profile = {r: profile[r] for r in useful}
    ranked = rank_by_utility(sub_profile, alpha=alpha, beta=beta, eps=eps)
    print(f"  [C] top-5 by utility: {[(r, round(u,2)) for r, u in ranked[:5]]}", flush=True)

    # ── Stage D: greedy set-cover ────────────────────────────────────────────
    base_selection, cover_trace = greedy_cover(sub_profile, D_star, alpha=alpha, beta=beta, eps=eps)
    print(f"  [D] greedy cover selected {len(base_selection)} rules", flush=True)

    # ── Stage F: k-fold stability (re-uses Stage A profile, LLM-free) ────────
    fold_results: list[dict] = []
    selection_freq = None
    stable_selection = base_selection
    if use_stability and len(all_doc_names) >= 4:
        fold_results, selection_freq = k_fold_selection(
            sub_profile, all_doc_names,
            k_folds=k_folds, fold_frac=fold_frac,
            alpha=alpha, beta=beta, eps=eps, seed=seed,
        )
        stable_selection = stable_rule_set(
            selection_freq, base_selection,
            k_folds=k_folds, min_folds=min_folds,
        )
        print(f"  [F] after k-fold (k={k_folds}, threshold≥{min_folds or (k_folds+1)//2}): "
              f"{len(stable_selection)} rules (was {len(base_selection)})", flush=True)

    # ── Stage E: merge verification (real LLM) + extension if below target ───
    print(f"  [E] verifying merge accuracy with {model_name}...", flush=True)
    t0 = time.time()
    final_selection, merge_acc, per_doc, tokens = verify_and_extend(
        stable_selection, ranked,
        documents, ground_truth, question, rule_folder, model_mod,
        target_accuracy=target_accuracy,
        max_extra_rules=max_extra_rules,
    )
    print(f"  [E] merge_acc={merge_acc:.2f} target={target_accuracy:.2f} "
          f"selected={len(final_selection)} ({time.time()-t0:.1f}s)", flush=True)

    # ── Stage G: optional budget pruning ─────────────────────────────────────
    pruned: list[dict] = []
    if rule_budget is not None and len(final_selection) > rule_budget:
        print(f"  [G] over budget ({len(final_selection)} > {rule_budget}), pruning...", flush=True)
        final_selection, merge_acc, per_doc, prune_tokens = _budget_prune(
            final_selection, documents, ground_truth, question, rule_folder, model_mod,
            target_accuracy=target_accuracy, budget=rule_budget,
        )
        tokens["qa_in"]     += prune_tokens["qa_in"]
        tokens["qa_out"]    += prune_tokens["qa_out"]
        tokens["judge_in"]  += prune_tokens["judge_in"]
        tokens["judge_out"] += prune_tokens["judge_out"]
        tokens["llm_calls"] += prune_tokens["llm_calls"]
        pruned = prune_tokens.get("pruned", [])

    # ── Enrich per_doc with token counts ──────────────────────────────────────
    from rule_refine.v1 import _count_tokens
    for entry in per_doc:
        dname = entry["doc_name"]
        doc = next((d for d in documents
                    if d.get("doc_name", d.get("origin", {}).get("filename", "")).replace(".pdf", "") == dname),
                   None)
        if doc:
            total = _count_tokens("\n".join(s.get("text", "") for s in doc.get("texts", [])))
            entry["total_doc_tokens"] = total
            entry["cost_ratio"] = round(entry["retrieved_tokens"] / total, 6) if total > 0 else 0.0

    # ── Copy selected rule files to output ───────────────────────────────────
    sel_dir = out_dir / question_slug
    sel_dir.mkdir(parents=True, exist_ok=True)
    for rn in final_selection:
        src = rule_folder / f"{rn}.py"
        if src.exists():
            shutil.copy2(src, sel_dir / f"{rn}.py")

    total_latency = round(time.time() - t_start, 3)

    # ── Build result ─────────────────────────────────────────────────────────
    result = {
        "question":               question,
        "question_slug":          question_slug,
        "timestamp":              timestamp,
        "algorithm":              "rule_refine_v2",
        "model":                  model_name,
        "num_documents":          len(documents),
        "all_rules_count":        len(profile),
        "useful_rules_count":     len(useful),
        "selected_rules_count":   len(final_selection),
        "selected_rules":         final_selection,
        "target_accuracy":        round(target_accuracy, 4),
        "proxy_ceiling":          round(proxy_ceiling, 4),
        "merge_accuracy":         round(merge_acc, 4),
        "D_star":                 sorted(D_star),
        "rule_profile":           _serialize_profile(profile),
        "ranked_by_utility":      [(r, round(u, 4)) for r, u in ranked],
        "set_cover_trace":        cover_trace,
        "fold_results":           fold_results,
        "selection_freq":         dict(selection_freq) if selection_freq else {},
        "base_selection":         base_selection,
        "stable_selection":       stable_selection,
        "budget_pruned":          pruned,
        "total_latency_seconds":  total_latency,
        "llm_calls":              tokens["llm_calls"],
        "qa_input_tokens":        tokens["qa_in"],
        "qa_output_tokens":       tokens["qa_out"],
        "judge_input_tokens":     tokens["judge_in"],
        "judge_output_tokens":    tokens["judge_out"],
        "total_input_tokens":     tokens["qa_in"] + tokens["judge_in"],
        "total_output_tokens":    tokens["qa_out"] + tokens["judge_out"],
        "verification_extras":    tokens.get("extras", []),
        "per_doc":                per_doc,
        "hyperparams": {
            "alpha":             alpha,
            "beta":              beta,
            "eps":               eps,
            "k_folds":           k_folds,
            "fold_frac":         fold_frac,
            "min_folds":         min_folds,
            "rule_budget":       rule_budget,
            "fuzzy_numeric":     fuzzy_numeric,
            "use_stability":     use_stability,
            "seed":              seed,
            "max_extra_rules":   max_extra_rules,
        },
    }

    # ── Write outputs ────────────────────────────────────────────────────────
    out_path = out_dir / f"{question_slug}_refine_v2.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_set_default),
                        encoding="utf-8")

    _update_summary(out_dir, result)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize_profile(profile: dict[str, dict]) -> dict[str, dict]:
    """Convert sets to sorted lists for JSON serialization."""
    return {
        r: {**p,
            "covered_docs": sorted(p["covered_docs"]),
            "proxy_docs":   sorted(p["proxy_docs"])}
        for r, p in profile.items()
    }


def _set_default(o):
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _empty_result(question, slug, ts, model, n_docs, all_rules, out_dir, t_start):
    return {
        "question":             question,
        "question_slug":        slug,
        "timestamp":            ts,
        "algorithm":            "rule_refine_v2",
        "model":                model,
        "num_documents":        n_docs,
        "all_rules_count":      len(all_rules),
        "selected_rules":       [],
        "merge_accuracy":       0.0,
        "D_star":               [],
        "note":                 "D* empty — proxy_judge found no rule retrieving GT in any doc.",
        "total_latency_seconds": round(time.time() - t_start, 3),
    }


def _update_summary(out_dir: Path, result: dict) -> None:
    summary_path = out_dir / "summary.json"
    summary: list[dict] = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = []

    entry = {
        "question":              result["question"],
        "question_slug":         result["question_slug"],
        "target_accuracy":       result.get("target_accuracy"),
        "merge_accuracy":        result.get("merge_accuracy"),
        "all_rules_count":       result.get("all_rules_count"),
        "selected_rules_count":  result.get("selected_rules_count"),
        "llm_calls":             result.get("llm_calls"),
        "total_latency_seconds": result.get("total_latency_seconds"),
    }

    updated = False
    for i, e in enumerate(summary):
        if e.get("question_slug") == result["question_slug"]:
            summary[i] = entry
            updated = True
            break
    if not updated:
        summary.append(entry)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _budget_prune(
    selection: list[str],
    documents: list[dict],
    ground_truth: dict,
    question: str,
    rule_folder: Path,
    model_mod,
    *,
    target_accuracy: float,
    budget: int,
) -> tuple[list[str], float, list[dict], dict]:
    """Stage G: drop rules one at a time whose removal preserves merge_acc ≥ target,
    until |selection| ≤ budget or no further safe removal exists."""
    from rule_refine.v1 import evaluate_merge_accuracy
    current = list(selection)
    tokens = {"qa_in": 0, "qa_out": 0, "judge_in": 0, "judge_out": 0, "llm_calls": 0, "pruned": []}

    last_per_doc: list[dict] = []
    last_acc = 0.0

    while len(current) > budget:
        best_removable = None
        best_acc = -1.0
        best_per_doc: list[dict] = []
        for r in current:
            trial = [x for x in current if x != r]
            acc, per_doc, qa_in, qa_out, j_in, j_out = evaluate_merge_accuracy(
                trial, documents, ground_truth, question, rule_folder, model_mod
            )
            tokens["qa_in"]     += qa_in
            tokens["qa_out"]    += qa_out
            tokens["judge_in"]  += j_in
            tokens["judge_out"] += j_out
            tokens["llm_calls"] += 2 * len(documents)
            if acc >= target_accuracy and acc > best_acc:
                best_acc = acc
                best_removable = r
                best_per_doc = per_doc

        if best_removable is None:
            break
        tokens["pruned"].append({"removed": best_removable, "acc_after": round(best_acc, 4)})
        current.remove(best_removable)
        last_acc = best_acc
        last_per_doc = best_per_doc

    if not last_per_doc:
        # No pruning happened; do one final eval so caller has fresh per_doc
        acc, last_per_doc, qa_in, qa_out, j_in, j_out = evaluate_merge_accuracy(
            current, documents, ground_truth, question, rule_folder, model_mod
        )
        tokens["qa_in"]     += qa_in
        tokens["qa_out"]    += qa_out
        tokens["judge_in"]  += j_in
        tokens["judge_out"] += j_out
        tokens["llm_calls"] += 2 * len(documents)
        last_acc = acc

    return current, last_acc, last_per_doc, tokens
