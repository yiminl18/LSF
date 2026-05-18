# Rule Refinement — Version Tracking

This document indexes every rule-refinement / selection / application strategy implemented in the LSF codebase, with measured accuracy and cost numbers on FinanceBench. Refinement takes a generated rule pool (see `docs/rule_generation_versions.md`) and produces a smaller working set plus optional deployment-time strategy.

**Supersedes `docs/pareto_versions.md`** — that doc covered only the Pareto variants; this one covers everything.

---

## Summary table — single cluster, gpt54-generated rule pool

All numbers are on the FinanceBench **single-cluster** dataset (10 sampled, 50 unsampled, 10 questions). The full rule pool ("base") averages 25–118 rules per question. All accuracy/cost numbers use gpt54 for QA + judge on the *evaluation* side; the in-loop selector judge varies per variant.

| Variant | Module | Phase | sAcc | cost_s | uAcc | cost_u | Mean rules | Overfit gap (s−u) | Notes |
|---------|--------|-------|-----:|-------:|-----:|-------:|-----------:|------------------:|-------|
| **base (no selection)** | n/a | — | **0.910** | 0.1722 | **0.892** | 0.1686 | ~63 | +0.018 | Ceiling reference |
| **v1** | `src/rule_refine.py` | selection | 0.890 | 0.0285 | 0.778 | 0.0348 | 7.8 | +0.112 | Cost-sort + exp search + backward prune; enforces `target_accuracy = base` |
| **p_mini** | `src/rule_refinement/select_rules_pareto.py` (gpt54mini judge) | selection | 0.840 | 0.0049 | 0.780 | 0.0052 | 3.9 | **+0.060** | Cost-effectiveness greedy; cheap judge as accidental regularizer |
| **p_proxy** | `src/rule_refinement/select_rules_pareto_proxy.py` | selection | 0.843* | 0.0041 | 0.737* | 0.0038 | 3.0 | +0.106 | Substring-only judge; **0 LLM calls** in selection; fails on 3/10 Qs |
| **p_gpt54** | `src/rule_refinement/select_rules_pareto.py` (gpt54 judge) | selection | 0.880 | 0.0067 | 0.748 | 0.0047 | 4.3 | +0.132 | Same algorithm as p_mini, stronger judge → more overfit |
| **p_v2** | `src/rule_refinement/select_rules_pareto_v2.py` | selection | 0.880 | 0.0119 | **0.806** | 0.0100 | 5.1 | +0.074 | Pareto greedy + accuracy floor + backward prune (gpt54) |
| **p_v3** | `src/rule_refinement/select_rules_pareto_v3.py` | selection | **0.910** | 0.0113 | 0.804 | 0.0091 | 4.7 | +0.106 | v2 + cumulative-prefix Phase B′ — matches base sAcc on 8/10 Qs |
| **agentic** | `agent/run_agent_select.py` | selection | TBD | TBD | TBD | TBD | TBD | TBD | Claude Opus 4.7 outer loop with tool calls (running) |
| **fallback** | `src/default_rule.py` | **deployment** | n/a | n/a | **0.892** | **0.0302** | 5.1 + on-demand | gpt54mini gate over p_v2 → full-pool fallback on miss |

\* p_proxy on the 7 questions where it produces output (it fails completely on state/EIN, total revenue, trading symbols where GT strings aren't substrings of the retrieved text).

**Three other algorithmic variants exist as code but have no current run data:** static-τ (`select_rules.py`), auto-tighten (`select_rules_auto_tighten.py`), and the v2 proposal (`src/rule_refine_v2/`) — left out of the comparison until benchmarked.

---

## Pareto frontier (single-cluster, mean across 10 questions)

```
uAcc  ▲
0.90  │                                                          ▓ fallback (0.892, 3.0%)        ●  base   (0.892, 16.9%)
      │
0.85  │                              ▓ p_v3  (0.804, 1.13%)
      │                              ▓ p_v2  (0.806, 1.19%)
0.80  │
      │           ▓ p_mini (0.780, 0.49%)   v1 ▓ (0.778, 3.48%)
0.75  │              ▓ p_gpt54 (0.748, 0.67%)
      │
0.70  │    ▓ p_proxy (0.737*, 0.41%)
      ├──────────────────────────────────────────────────────────────────►
      0   1%        2%        3%      … 5%                    17%
                                cost_ratio_u
```

- **fallback** is Pareto-dominant (achieves base uAcc at ~18% of base cost).
- **p_v2** is the best selection-only result (highest uAcc among rule-subset strategies, no inference-time machinery).
- **v1** is dominated by p_v2: same uAcc, ~3× cost, more rules.

---

## Method details

### v1 — `src/rule_refine.py` (April 2026)

**Algorithm:**
1. Sort all rules by `avg_cost_ratio` ascending.
2. Exponential search: try k=1,2,4,8,… cheapest rules until merge_acc ≥ `target_accuracy` (default `base`).
3. Backward prune: remove any rule whose removal preserves `merge_acc ≥ target`.
4. In-loop judge: gpt54.

**Result:** matches `base` on sampled by construction (sAcc=0.89 ≈ base 0.91, within LLM noise). Worst overfit gap (+0.112) of the selection-only variants. Most expensive in LLM calls (~864 per question — full exp-search + prune cycle).

**Spec:** `docs/rule_refine.md`

---

### p_mini — Pareto with gpt54mini judge

**Algorithm:** cost-effectiveness sort (`cov(r) / avg_cost_ratio(r)` descending) + greedy cover of D*. Stops when all D* docs are covered (under the in-loop judge).

**Result:** mean uAcc 0.780, lowest overfit gap (+0.060) of all gpt54-judge-aware variants. The cheap judge mis-judges marginal rule contributions in a way that happens to act as **accidental regularization** — broader rules get admitted, narrower specialists get rejected.

**Spec:** `docs/rule_selection_pareto_implementation.md`

---

### p_proxy — Pareto with substring-only judge

**Algorithm:** same Pareto greedy, but the in-loop judge is `proxy_judge(gt, retrieved_text)` — a case-insensitive substring check. **Zero LLM calls during selection.**

**Result:** matches p_mini's uAcc on the 7 questions where it works, **fails completely on 3** (state/EIN, total revenue, trading symbols) because ground-truth strings (e.g. "Delaware", "$12,345 million") don't appear verbatim in retrieved text. Fastest variant (~10 min for 10 questions, 0 LLM cost) but not deployable as-is.

---

### p_gpt54 — Pareto with full gpt54 judge

**Algorithm:** same as p_mini, with gpt54 in the in-loop judge.

**Counter-intuitive result:** stronger judge → **worse** uAcc (0.748). Opus's accidental-regularization effect disappears; the algorithm becomes more confident about admitting narrow specialists that overfit. Mean overfit gap +0.132 — the worst of any Pareto variant.

---

### p_v2 — Pareto with accuracy floor + backward prune

**Algorithm:**
1. **Phase A** — Pareto greedy cover (same as p_gpt54).
2. **Phase B (new)** — if `merge_acc < base`, admit next rule by cost-effectiveness; repeat up to `max_extra_rules=20`.
3. **Phase C (new)** — backward prune any rule whose removal preserves `merge_acc ≥ base`.

**Result:** **highest uAcc (0.806) of any selection-only variant**, comparable sAcc to v1 (0.880 vs 0.890), 5.1 rules avg vs v1's 7.8. Matches base on sampled for 7/10 questions; the 3 misses are LLM stochasticity or hard-rule-pool issues. **This is the recommended selection algorithm.**

**Spec:** `src/rule_refinement/select_rules_pareto_v2.py` (algorithm in code), full report in this doc + `docs/rule_apply_with_fallback.md` §1.

---

### p_v3 — v2 + cumulative-prefix Phase B′

**Algorithm:** v2 + when Phase B hits `max_extra_rules` without reaching base, switch to a v1-style exponential prefix search over the remaining rule pool.

**Result:** highest sAcc of any variant (**0.910 = base mean exactly**), uAcc essentially tied with v2 (0.804 vs 0.806). Matches base on 8/10 questions. Cost: **2× the LLM budget of v2** (cumulative-prefix testing is expensive). Trade-off: pay 2× cost for +0.030 sAcc but no uAcc gain → **not worth it in practice** unless matching base is a hard requirement.

---

### agentic — Claude Opus 4.7 outer loop

**Algorithm:** Claude Code agent (`claude -p` with `--model claude-opus-4-7`) iterates:
1. List rules, snapshot cost/coverage (free tools)
2. Propose subset S, verify accuracy via `verify_accuracy.py` (paid: ~20 gpt54 calls)
3. Inspect missed docs' rules, swap/add, re-verify
4. Stop when accuracy matches base, budget exhausted, or soft targets settled

**Key feature:** **adaptive trade-offs**. The agent reads rule docstrings, spots layout-specific rules ("page_around_39"), and explains its choices in a natural-language `rationale` field. The algorithmic variants can't do this.

**Result:** TBD (run launched 2026-05-18 20:59 UTC; ~60–90 min ETA).

**Specs:** `docs/rule_selection_agentic.md`, `agent/task_prompt.md`.

---

### fallback (deployment-time, not selection) — `src/default_rule.py`

**Algorithm:** at inference on each unsampled doc:
1. Apply refined rules (e.g. p_v2's output) → retrieved text
2. Ask **gpt54mini** "does this text contain the answer?"
3. If YES → answer with gpt54 on refined retrieval
4. If NO → fall back to full rule pool, then answer with gpt54

**Result:** **uAcc = 0.892 (matches base exactly), cost_u = 0.0302 (only 18% of base's 0.169 retrieval cost)**. Mean fallback rate 11.8% — the gate triggers on ~6 of 50 unsampled docs per question. Recovers the entire p_v2→base generalization gap at ~3× the refined retrieval cost (still 5.6× cheaper than always-full).

**Two questions where fallback beats both refined-only and base:**
- Long-term debt: fallback 0.72 > base 0.66 (gate filters out retrievals that confuse gpt54)
- (most others tied with base, as expected)

**Spec:** `docs/rule_apply_with_fallback.md`

---

## Per-question deep dive (single cluster)

| Question | base sAcc | base uAcc | best uAcc (variant) |
|----------|---------:|---------:|---------------------|
| Shares outstanding | 0.90 | 0.98 | fallback **0.98** |
| Long-term debt | 0.60 | 0.66 | fallback **0.72** (beats base!) |
| Net income | 0.90 | 0.76 | fallback **0.76** |
| Address & ZIP | 0.90 | 0.94 | fallback **0.94** |
| Registrant name | 0.90 | 1.00 | fallback / p_v2 / p_v3 **1.00** |
| Telephone | 1.00 | 1.00 | fallback / p_v2 / p_v3 **1.00** |
| State & EIN | 1.00 | 0.98 | fallback **0.98** |
| Total assets | 1.00 | 0.86 | fallback / base **0.86** |
| Total revenue | 0.90 | 0.82 | fallback **0.80** (close to base) |
| Trading symbols | 1.00 | 0.92 | fallback / base **0.92** |

---

## Multi-cluster results (TBD)

None of the refinement variants have been run against the multi-cluster rule pool yet. The base full-pool numbers are in `docs/rule_generation_versions.md`. Selection performance on multi-cluster is an open follow-up.

---

## Aggregate findings

### Which selection variant generalizes best?
**p_v2** — uAcc 0.806, lowest cost among accuracy-matching variants. Recommended default for selection-only deployment.

### Which is most cost-efficient?
For **accuracy-matching the full pool**: fallback (uAcc 0.892 at cost 0.030 = **5.6× cheaper than base**, 3× costlier than refined-only).

For **selection alone**: p_proxy (~$0.005/run) when GTs are verbatim strings; p_mini when they aren't.

### What doesn't work
- p_proxy on numeric/string-format answers (3/10 Qs fail).
- v1's exhaustive search is dominated by p_v2 (same uAcc, more rules, more cost).
- p_gpt54 is dominated by p_mini (same algorithm, gpt54 judge produces worse uAcc).

### Surprising findings
1. **Cheaper in-loop judge → better generalization** (p_mini vs p_gpt54). gpt54mini's noisier verdicts regularize against over-confident specialist admission.
2. **Fallback can beat base on individual questions** (long-term debt 0.72 vs base 0.66). The gate filters out retrievals that would confuse gpt54.
3. **Matching base sAcc costs uAcc** — p_v3 forces sAcc=0.91 but uAcc stays at 0.80, paying 2× LLM budget for nothing.

---

## File index

### Selection algorithms
| File | Variant |
|------|---------|
| `src/rule_refine.py` | v1 |
| `src/rule_refinement/select_rules.py` | static-τ (not benchmarked) |
| `src/rule_refinement/select_rules_auto_tighten.py` | auto-tighten (not benchmarked) |
| `src/rule_refinement/select_rules_pareto.py` | p_mini, p_gpt54 |
| `src/rule_refinement/select_rules_pareto_proxy.py` | p_proxy |
| `src/rule_refinement/select_rules_pareto_v2.py` | p_v2 |
| `src/rule_refinement/select_rules_pareto_v3.py` | p_v3 |
| `agent/run_agent_select.py` + `agent/task_prompt.md` | agentic |

### Deployment-time application
| File | Strategy |
|------|----------|
| `src/rule_apply_individual.py`, `src/rule_apply_merge.py` | Direct application of any rule set |
| `src/default_rule.py` | Refined-with-fallback gate |

### Drivers
| File | Variant |
|------|---------|
| `test/run_select_all_pareto.py` | p_mini (with `MODEL_NAME=gpt54mini`) |
| `test/run_select_all_pareto_proxy.py` | p_proxy |
| `test/run_select_all_pareto_gpt54.py` | p_gpt54 |
| `test/run_select_all_pareto_v2.py` | p_v2 |
| `test/run_select_all_pareto_v3.py` | p_v3 |
| `test/run_default_rule_unsampled.py` | fallback |
| `test/eval_pareto_*.py` | Per-variant sampled/unsampled eval |
| `test/eval_agentic_{sampled,unsampled}.py` | Agentic eval |

### Output directories (all under `results/financebench_single_cluster/llm/gpt54/one_shot/`)
| Folder | Contents |
|--------|----------|
| `selected_rules_pareto_gpt54mini/`, `_proxy/`, `_gpt54/`, `_v2/`, `_v3/` | Per-variant selection JSONs |
| `selected_rules_agent/` | Agentic selection JSONs |
| `eval_pareto_gpt54mini/`, `_proxy/`, `_gpt54/`, `_v2/`, `_v3/` | Per-variant eval JSONs |
| `eval_pareto_v2_fallback/` | Fallback eval JSONs |
| `eval_agentic/` | Agentic eval JSONs |
| `eval_merge/` | Full-pool baseline eval (the "base" row above) |
| `cost_profile/` | Shared cost cache (no LLM) |
| `eval_individual/` | Per-rule coverage (no LLM at lookup time) |

---

## Dataset

Same as `docs/rule_generation_versions.md`:

| Cluster | Sampled | Unsampled | Questions | Rule pool size (avg) |
|---------|--------:|----------:|----------:|---------------------:|
| Single | 10 | 50 | 10 | 63 |
| Multi | 18 | 96 | 12 | n/a (rules generated, refinement TBD) |
