# Pareto-Frontier Rule Selection — Implementation Notes

This document maps the Pareto-frontier rule selection algorithm (see `rule_selection_pareto.pdf`) to concrete modules, scripts, and on-disk artefacts in the LSF codebase. It is the companion to `rule_selection_implementation.md`, which documents the static-`&tau;` and auto-tighten pipelines. The Pareto-frontier pipeline is additive — none of the existing modules under `src/rule_refinement/` is overwritten.

> Paths are written relative to the LSF repo root (`~/Documents/Codebase/LSF/`). The single-cluster, LLM-generated, one-shot variant is used throughout for examples; the agent-generated paths follow the same structure.

---

## 1. One-line intuition

Pick rules with **good** (not highest) coverage at **low** (not lowest) cost — the cost-effective sweet spot, not either extreme. Operationally this means: replace the cost-ascending sort with a `cov(r) / W_r` descending sort, record a breakpoint after every admission, and stop when `D*` is fully covered. The output is the full cost-vs-accuracy curve, not a single selected set.

---

## 2. Mapping: algorithm symbol &rarr; code

The notation is the same as `rule_selection_implementation.md` §1, with a few additions specific to the frontier formulation.

| Symbol | Meaning | Where it lives |
|---|---|---|
| `r` &isin; `R` | A single rule | `rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm/rule_<name>.py`. |
| `d` &isin; `D` | A single document | `data/financebench/processing/<DOC_NAME>_reconstructed.json`. |
| `c(r, d)` | Per-(rule, doc) token cost ratio | `cost_profile/<slug>.json::per_doc_ratio` (free, no LLM). |
| `W_r` | Total rule cost `Σ_d c(r,d)` | `cost_profile/<slug>.json::avg_cost_ratio` multiplied by `m`; sorting on either is equivalent. |
| `cov(r)` | Per-rule coverage = mean of `a(r, ·)` | Top-level `accuracy` in `eval_individual/<slug>_10_llm/<r>_eval.json`. |
| `priority(r)` | **NEW** — cost-effectiveness ratio | `cov(r) / (avg_cost_ratio(r) + ε)`; computed at sort time, not stored. |
| `A(S, d)` | Merge accuracy of subset `S` on doc `d` | Output of `src/rule_apply_merge.py::rule_apply_merge` + `src/rule_refinement/eval_judge.py::judge` under `M_prod` = `gpt54`. |
| `A*(d)` | Merge accuracy of full rule set | `eval_merge/<slug>_10_sampled.json::per_doc[*].correct`. |
| `D*` | Docs the full set solves | Set extracted from the same `per_doc` list. |
| `α(S)` | **NEW** — fractional accuracy match | `|covered ∩ D*| / |D*|`, computed incrementally as `_greedy_cover` admits rules. |
| `F` | **NEW** — Pareto frontier | List of breakpoints `(W(S_i), α(S_i), rules_admitted_so_far)` recorded one per admission. |

---

## 3. What's already implemented (the primitives we reuse, unchanged)

Every primitive the Pareto pipeline needs is already present under `src/rule_refinement/`. Nothing in this file is modified by the Pareto variant.

| Primitive | Where | What it does |
|---|---|---|
| Cost profile (Phase 0) | `src/rule_refinement/cost_profile.py::compute_rule_costs(rules_dir, doc_paths)` and `load_or_compute_cost_profile(... cache_path)` | Returns `{rule_name: {"W", "per_doc", "per_doc_ratio", "avg_cost_ratio"}}`. No LLM calls. Cached at `results/.../cost_profile/<slug>.json`. |
| Baseline targets (Phase 1) | `src/rule_refinement/baseline_targets.py::load_target_docs(eval_merge_path)` | Reads `eval_merge/<slug>_sampled.json::per_doc[*].correct == True`. No new LLM calls. |
| Per-rule coverage source | `src/rule_refinement/coverage_check.py::load_or_compute_coverage(rule_name, eval_individual_path, fallback)` | Reads `accuracy` from `eval_individual/<slug>_10_llm/<r>_eval.json` if present; falls back to `0.0` (or a user-supplied closure) otherwise. |
| Greedy cover (Phase 2) | `src/rule_refinement/select_rules.py::_greedy_cover(rules_sorted, target_docs, ...)` | Walks `rules_sorted` in given order; admits each rule that gains at least one doc; returns `(newly_selected, per_rule_gained, qa_calls, judge_calls, in_tok, out_tok)`. **Already sort-agnostic** — the priority is determined by how `rules_sorted` is built by the caller. |
| Judge primitives | `src/rule_refinement/eval_judge.py::judge(question, gt, predicted, model_name)` and `proxy_judge(gt, retrieved_text)` | LLM-as-judge under `M_prod` (returns `(correct, in_tok, out_tok)`) and substring pre-filter. |
| Merge eval primitive | `src/rule_apply_merge.py::rule_apply_merge(document, rule_names, ...)` | Applies the union, runs the QA LLM, returns prediction + accounting. |

The critical observation: **`_greedy_cover` is already sort-agnostic.** It accepts a pre-sorted `rules_sorted` list and walks it. The Pareto variant changes nothing inside `_greedy_cover`; it just builds `rules_sorted` differently and records breakpoints around each admission.

---

## 4. Phase-by-phase implementation plan

### Phase 0 — Cost profile (reuse `cost_profile.py`)

No change. Call `load_or_compute_cost_profile(rules_dir, doc_paths, cache_path)` exactly as the static and auto-tighten drivers do. Result is cached at `results/.../cost_profile/<slug>.json` and reused across all three pipelines.

### Phase 0.5 — Coverage source (reuse `coverage_check.py`)

The Pareto sort key `priority(r) = cov(r) / W_r` requires `cov(r)` for **every** rule, not just for selected ones. There are two ways to populate it:

1. **Read from `eval_individual/<slug>_10_llm/`** (current production path). Already produced by `test/run_eval_individual.py` using `M_prod`. Free at selection time.
2. **Run a cheap-LLM coverage sweep** (proposed, not yet in the codebase). Mirror `test/run_eval_individual.py` but pass `model_name="gpt54mini"`; write to `results/.../eval_individual_cheap/<slug>_10_llm/`. The `coverage_check.load_or_compute_coverage` API already accepts a path, so swapping the source is one line.

Either way, build the per-rule coverage map exactly the way `select_rules_auto_tighten.py` already does (lines 187–190):

```python
cov_map = {
    r: load_or_compute_coverage(
        r,
        eval_individual_dir / f"{r}_eval.json",
        lambda: 0.0,
    )
    for r in cost_profile
}
```

### Phase 1 — Baseline (reuse `baseline_targets.py`)

No change. `load_target_docs(eval_merge_path)` returns `D*` from `eval_merge/<slug>_sampled.json`.

### Phase 2 — Cost-effectiveness greedy with frontier recording

This is the only new piece. Sort by `cov / cost` descending, call the existing `_greedy_cover` once, and record a frontier breakpoint around every admitted rule.

```python
priority = lambda r: cov_map.get(r, 0.0) / (
    cost_profile[r]["avg_cost_ratio"] + 1e-9
)
rules_sorted = sorted(cost_profile.keys(), key=priority, reverse=True)

# Call the existing _greedy_cover unchanged.
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
```

The frontier itself is reconstructed from the admission order `S` and `per_rule_gained` (no extra LLM calls — both are returned by `_greedy_cover`):

```python
frontier = [{"cost": 0.0, "accuracy_match": 0.0, "rules_admitted": []}]
running_cost = 0.0
running_covered: set[str] = set()
for r in S:
    running_cost += cost_profile[r]["avg_cost_ratio"]
    running_covered |= per_rule_gained[r]
    frontier.append({
        "cost": round(running_cost, 6),
        "accuracy_match": round(len(running_covered) / len(target_docs), 4),
        "rules_admitted": list(S[: S.index(r) + 1]),
        "added_rule": r,
        "rule_cov": round(cov_map.get(r, 0.0), 4),
        "rule_avg_cost": round(cost_profile[r]["avg_cost_ratio"], 6),
    })
```

Termination: `_greedy_cover` already stops as soon as `U = ∅`, which corresponds to `α(S) = 1`. If you want to continue admitting marginal rules past full cover for robustness, wrap `_greedy_cover` accordingly — not needed for the standard frontier.

### Phase 3 (optional) — Hard `τ_floor` safety filter

The Pareto frontier formulation does not require a `τ` parameter, but a safety filter is cheap and harmless if you want one. Reuse `coverage_check.filter_by_tau(...)` with a low `τ_floor` (default 0.10) on the final `S`, then return any banned-rule docs to `U` and resume `_greedy_cover` on `rules_sorted` from the next-best candidate — exactly mirroring `select_rules.py` lines 199–240.

### Phase 4 (optional) — Knee-point and query-table precomputation

After the frontier is built, compute the operator-facing summary:

```python
def make_query_table(frontier, thresholds=(0.5, 0.8, 0.9, 0.95, 1.0)):
    out = {}
    for t in thresholds:
        match = next((p for p in frontier if p["accuracy_match"] >= t), None)
        out[f"{t:.2f}"] = (
            {"cost": match["cost"], "n_rules": len(match["rules_admitted"])}
            if match else None
        )
    return out

def make_knee_point(frontier, lam=10.0):
    # argmax_p (accuracy_match - lam * cost)
    return max(frontier, key=lambda p: p["accuracy_match"] - lam * p["cost"])
```

The `lam` hyperparameter controls the trade-off slope; a single value is fine because operators read off whatever row they actually want from the `query_table` anyway.

---

## 5. Files to add (additive — no overwrites)

```
src/rule_refinement/
  select_rules_pareto.py             # NEW — exports run_selection_pareto(...).
                                     # Imports _greedy_cover from select_rules.py
                                     #         load_target_docs from baseline_targets.py
                                     #         load_or_compute_coverage from coverage_check.py
                                     #         load_or_compute_cost_profile from cost_profile.py

test/
  run_select_all_pareto.py           # NEW — driver mirroring run_select_all.py and
                                     # run_select_all_auto_tighten.py; writes to
                                     # results/.../selected_rules_pareto/.

results/financebench_single_cluster/llm/gpt54/one_shot/
  cost_profile/<slug>_10_llm.json    # SHARED — same cache as static / auto-tighten.
  selector_run_pareto/<slug>/...     # NEW — merge prediction outputs from Phase 2.
  selected_rules_pareto/<slug>.json  # NEW — frontier + query table.
```

Output schema for `selected_rules_pareto/<slug>.json`:

```json
{
  "question": "...",
  "question_slug": "...",
  "mode": "pareto",
  "baseline_accuracy": 0.90,
  "selected_rules": ["rule_a", "rule_b", ...],
  "selected_avg_cost_ratio_sum": 0.0080,
  "covered_docs": ["..."],
  "uncovered_docs": [],
  "selector_accuracy": 1.00,
  "frontier": [
    {"cost": 0.0000, "accuracy_match": 0.00, "rules_admitted": [], "added_rule": null, "rule_cov": null, "rule_avg_cost": null},
    {"cost": 0.0015, "accuracy_match": 0.50, "rules_admitted": ["rule_a"], "added_rule": "rule_a", "rule_cov": 0.90, "rule_avg_cost": 0.0015},
    {"cost": 0.0045, "accuracy_match": 0.80, "rules_admitted": ["rule_a", "rule_b"], "added_rule": "rule_b", "rule_cov": 0.70, "rule_avg_cost": 0.0030},
    {"cost": 0.0080, "accuracy_match": 1.00, "rules_admitted": ["rule_a", "rule_b", "rule_c"], "added_rule": "rule_c", "rule_cov": 0.60, "rule_avg_cost": 0.0035}
  ],
  "query_table": {
    "0.50": {"cost": 0.0015, "n_rules": 1},
    "0.80": {"cost": 0.0045, "n_rules": 2},
    "0.95": {"cost": 0.0080, "n_rules": 3},
    "1.00": {"cost": 0.0080, "n_rules": 3}
  },
  "knee_point": {"cost": 0.0045, "accuracy_match": 0.80, "n_rules": 2},
  "llm_calls": {"phase_2_incremental": 14, "phase_2_judge": 14},
  "token_usage": {"total_input_tokens": ..., "total_output_tokens": ...}
}
```

---

## 6. End-to-end recipe

Both the upstream prerequisites and the Phase 0 cache are shared with the other two pipelines.

```bash
# Prereqs (existing pipeline — needed by all three modes):
python test/run_eval_merge_sampled.py    # eval_merge/<slug>_sampled.json (A* and D*)
python test/run_eval_individual.py       # eval_individual/<slug>/<r>_eval.json (cov)

# Three coexisting modes:
python test/run_select_all.py                  # static τ_floor → selected_rules/
python test/run_select_all_auto_tighten.py     # auto-tighten   → selected_rules_auto/
python test/run_select_all_pareto.py           # Pareto         → selected_rules_pareto/   (NEW)
```

For a single-question dry run:

```python
from rule_refinement.select_rules_pareto import run_selection_pareto
result = run_selection_pareto(
    rules_dir=RULES_BASE_DIR,
    eval_merge_path=eval_merge_path,
    eval_individual_dir=eval_individual_dir,
    documents=doc_map,
    question_slug=rule_slug,
    question=question,
    labels=labels,
    cost_profile=cost_profile,
    model_name="gpt54",
    output_dir=SELECTOR_RUN_PARETO_DIR,
    tau_safety_floor=0.10,   # optional; default 0
)
# result["frontier"] is the curve; result["query_table"] is the lookup.
```

---

## 7. Comparison to existing pipelines

| Aspect | Static τ (`select_rules.py`) | Auto-tighten (`select_rules_auto_tighten.py`) | Pareto (`select_rules_pareto.py`, NEW) |
|---|---|---|---|
| Sort key in `_greedy_cover` | `avg_cost_ratio` ascending | `avg_cost_ratio` ascending | `cov(r) / avg_cost_ratio` descending |
| Coverage source | `eval_individual/<slug>/<r>_eval.json::accuracy` (Phase 3 lookup on `S` only) | Same, but pre-loaded into `cov_map` for the full pool (line 187) | Same, but used **at sort time** for the full pool |
| Hyperparameter | `tau` (default 0.20) | `tau_floor` + auto-tightening (default 0.20) | None required; optional `tau_safety_floor` (default 0) |
| Greedy bias | Cheap-narrow rules (per overfit report) | Cheap-narrow then ban-and-resume + backward prune | Cost-effective broad rules first |
| Output | One selected set + `tau` | One selected set + `tau_floor`, `tau_best`, `tightening_history` | Frontier curve + `query_table` + `knee_point` |
| Early stop | First feasible cover of `D*` | Same, then Phase 3.5 prune + Phase 4 tighten | First feasible cover of `D*` (no tighten needed) |
| Output folder | `selected_rules/` | `selected_rules_auto/` | `selected_rules_pareto/` |
| Merge-eval folder | `selector_run/` | `selector_run_auto/` | `selector_run_pareto/` |

The static and auto-tighten modes solve point optimisations on the same frontier — the Pareto pipeline subsumes both as queries on its output:

- Static answer ≈ row at `accuracy_match = 1.00` in the Pareto `query_table`.
- Auto-tighten's `tau_best` is the minimum `rule_cov` across `selected_rules` at the same row.

So Pareto can also serve as a diagnostic: if its frontier disagrees with the static or auto-tighten output on the same question, the disagreement is informative (typically the Pareto frontier finds a much cheaper full-coverage point because cost-effectiveness sorting reaches the broad rules earlier).

---

## 8. LLM-call accounting

Let `m = |D*|`, `n = |R|`, `k = |S|`.

| Phase | LLM calls | Notes |
|---|---|---|
| 0. Cost profile | 0 | tiktoken only |
| 0.5. Coverage source | 0 if `eval_individual` cached, else `n · m` (M_prod) or `n · m` (M_cheap) | Same as auto-tighten |
| 1. Baseline | 0 (disk read) | `eval_merge` already populated by the existing pipeline |
| 2. Greedy + frontier recording | `Σ u_i` ≤ `n · m`, in practice 30–80 per question | Identical to static-`τ` Phase 2 in volume; cheaper in practice because cost-effective sort reaches the cover faster |
| 3 (optional) τ_safety_floor | `k · m` if it bans anything | Often zero with low default `τ_safety_floor` |

Total per question: roughly `m + (Σ u_i)` `M_prod` calls — same as static-`τ`, with no auto-tighten Phase 4 overhead.

---

## 9. Sanity checks before trusting a frontier

- **Monotonicity.** Assert `frontier` is non-decreasing in both `cost` and `accuracy_match`. (`accuracy_match` should be strictly increasing across admissions; equal-`accuracy` admissions should not exist by `_greedy_cover`'s "must gain a doc" rule.)
- **Endpoints.** First breakpoint `(0, 0)`. Last breakpoint `(W(S), 1.0)` whenever `D*` is fully covered.
- **Cross-check with static τ.** The static-`τ` selector's `selected_avg_cost_ratio_sum` must be ≥ the Pareto frontier's full-coverage cost. If it's lower, something is wrong with the cov_map or the merge-eval cache.
- **Accuracy invariant.** Rerun `rule_apply_merge(rule_names=S)` + judge on every `d ∈ D*`. Every doc must come back `correct = True`. This is the hard contract the algorithm promises and must be re-verified at the end.
- **Knee detectability.** If `knee_point["accuracy_match"]` equals `1.0`, the curve has no knee (every admission is necessary). If it's below 1.0, the curve has a discernible elbow — operators may want to deploy at the knee.

---

## 10. Open implementation notes

- **Sort-key stability for ties.** When two rules have equal `cov / cost` ratio, break ties by `cov` descending then `cost` ascending. This favours broad-and-cheap over narrow-and-cheap at the boundary.
- **Cov source choice.** Use `M_prod` coverage if available; otherwise the cheap-LLM coverage sweep (see §3 of `rule_selection_pareto.pdf` and the `coverage_check.load_or_compute_coverage` API). Calibration: Spearman correlation between `cov_cheap` and `cov_prod` on 20 rules. Target ≥ 0.85.
- **Frontier reproducibility.** `_greedy_cover` already writes per-(question_slug, rule_set_slug) merge predictions to `selector_run_pareto/<slug>/`. Replaying a run with the same `cov_map` and `cost_profile` should reproduce the frontier exactly.
- **Cross-question summary.** Once `selected_rules_pareto/summary.json` is produced (a list of per-question frontiers), the corpus-level guarantee is `min_q max_p {p["accuracy_match"] : p["cost"] ≤ C}` for a chosen total-cost budget `C` — the worst-case accuracy across questions at cost `C`. This statistic is impossible to obtain from the static or auto-tighten outputs.
- **Integration with `polish_rule_set.py`.** The existing drop/swap polish is unnecessary in Pareto mode because cost-effectiveness sorting already avoids the rule-bloat pathology that polish exists to fix. Skip it by default; re-enable only for diagnostic comparison runs.
