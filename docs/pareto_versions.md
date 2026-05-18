# Pareto Rule Selection — Version Tracking

> **⚠️ Superseded by `docs/rule_refinement_versions.md`** — that doc covers Pareto plus v1, agentic, and the fallback strategy with up-to-date numbers. Keep this file only as the Pareto-specific history.


This document tracks all Pareto-frontier rule selection variants implemented in `src/rule_refinement/`. Each variant is a separate file (additive — none overwrites another) with its own drivers and output folders so they can be compared head-to-head.

Companion to `docs/rule_selection_pareto_implementation.md` (which specifies the original Pareto algorithm).

---

## Comparison summary

Measured on FinanceBench 10 sampled docs + 50 unsampled docs.

| Variant | Module | Driver | Output folder | Selection model | In-loop judge | sAcc | uAcc | Overfit | Mean rules | Wall time |
|---------|--------|--------|---------------|-----------------|----------------|-----:|-----:|--------:|------------:|----------:|
| **p_mini** | `select_rules_pareto.py` | `run_select_all_pareto.py` | `selected_rules_pareto_gpt54mini/` | gpt54mini | gpt54mini LLM | 0.840 | 0.780 | +0.060 | 3.9 | ~37 min |
| **p_proxy** | `select_rules_pareto_proxy.py` | `run_select_all_pareto_proxy.py` | `selected_rules_pareto_proxy/` | none | substring proxy | 0.843* | 0.737* | +0.106* | 3.0 | **~10 min** |
| **p_gpt54** | `select_rules_pareto.py` | `run_select_all_pareto_gpt54.py` | `selected_rules_pareto_gpt54/` | gpt54 | gpt54 LLM | 0.880 | 0.748 | +0.132 | 4.3 | ~50 min |
| **p_v2** | `select_rules_pareto_v2.py` | `run_select_all_pareto_v2.py` | `selected_rules_pareto_v2/` | gpt54 | gpt54 LLM | 0.880 | **0.806** | +0.074 | 5.1 | ~66 min |
| **p_v3** | `select_rules_pareto_v3.py` | `run_select_all_pareto_v3.py` | `selected_rules_pareto_v3/` | gpt54 | gpt54 LLM | TBD | TBD | TBD | TBD | TBD |

\* p_proxy on 7/10 questions only — failed completely (0 rules selected) on state/EIN, total revenue, trading symbols because GT strings don't appear verbatim. Mean on all 10 (0-rule treated as 0): sAcc 0.590, uAcc 0.516.

### Baselines for context

| Reference | sAcc | uAcc | Overfit | Mean rules |
|-----------|-----:|-----:|--------:|----------:|
| v1 (cost-sort + prune, in `src/rule_refine.py`) | 0.890 | 0.778 | +0.112 | 7.8 |

---

## p_mini — original Pareto with gpt54mini judge

**Algorithm** (see `docs/rule_selection_pareto_implementation.md`):
1. Compute `cov(r) / avg_cost_ratio(r)` descending sort of all rules.
2. Greedy cover: walk sorted list, admit rule `r` iff `working_S ∪ {r}` makes at least one uncovered doc correct under gpt54mini judge.
3. Record frontier breakpoint per admission; build `query_table` and `knee_point`.
4. No accuracy-floor enforcement, no backward pruning.

**Key property:** in-loop judge is gpt54mini (cheap). This judge happens to be noisier than gpt54 — acting as accidental regularization. Often misses what gpt54 would call correct.

**Wins / losses vs v1:**
- ✅ Mean overfit halved: 0.112 → 0.060 (best generalization at the time)
- ✅ Fewer rules per Q (3.9 vs 7.8)
- ❌ Doesn't match `base` on sampled: mean sAcc 0.84 vs base 0.91

---

## p_proxy — substring-only judge (no LLM in selection loop)

**Algorithm:** same as p_mini except the in-loop judge is replaced with `proxy_judge` from `eval_judge.py` — a case-insensitive substring check `gt.lower() in retrieved_text.lower()`. **Zero LLM calls during selection.**

**Key property:** the selection criterion exactly aligns with what gpt54 needs at eval time (the GT string must be retrievable). For string-answer questions this works cleanly.

**Wins / losses vs p_mini:**
- ✅ 3.7× faster (10 min vs 37 min); zero LLM cost during selection
- ✅ Wins big on shares outstanding (+0.20 sAcc), net income (+0.20 sAcc) — strictness rejects rules where gpt54mini was hallucinating agreement
- ❌ **Catastrophic failure on 3 questions** (0 rules selected) where GT strings never appear verbatim:
    - state & EIN — GT "Delaware" but doc says "DE"
    - total revenue — GT "$12,345 million" but doc says "12,345"
    - trading symbols — formatting variants on tickers

**Recommended use:** as a fast pre-filter, or with a fuzzy numeric proxy + LLM fallback for failing questions.

---

## p_gpt54 — Pareto with gpt54 judge

**Algorithm:** identical to p_mini but in-loop judge is gpt54 (production model). Same driver, just `MODEL_NAME = "gpt54"` flipped.

**Motivation:** test whether the model-disagreement gap between cheap selector judge and production evaluator was driving p_mini's overfit improvement.

**Key finding (counter-intuitive):**
- Mean sAcc improves: 0.840 → 0.880 (closer to base on sampled)
- Mean uAcc *drops*: 0.780 → 0.748
- Overfit gap *doubles*: 0.060 → 0.132 — same as v1!

**Interpretation:** the gpt54mini judge's noisy verdicts acted as accidental regularization, preventing the algorithm from confidently admitting narrow-but-overfit "specialist" rules. With a more confident gpt54 judge, the algorithm becomes more aggressive and overfits — same as v1.

**Concrete failure:** long-term debt sAcc=0.60 / uAcc=**0.00** (gap +0.60). The 2 rules gpt54 picked fire on all sampled docs but on zero unsampled docs.

---

## p_v2 — gpt54 selection with accuracy floor + backward pruning

**Algorithm** (see `src/rule_refinement/select_rules_pareto_v2.py`):
1. **Phase A:** cost-effectiveness greedy cover (same as p_gpt54).
2. **Phase B (acc floor, new):** if `merge_acc(S) < base`, iteratively add the next rule from the cost-effectiveness ordering and re-eval. Repeat up to `max_extra_rules=20`.
3. **Phase C (backward prune, new):** for each rule in S (last-added first), test removal. Drop iff `merge_acc ≥ base` remains true.
4. Frontier, query_table, knee_point built from final S.

**Motivation:** v1 always matches base on sampled because it explicitly targets accuracy. Pareto p_mini and p_gpt54 don't (they just cover D\* under their judge). v2 grafts v1's accuracy guarantee onto Pareto's cost-effectiveness sort.

**Key findings:**
- ✅ **Best uAcc of any variant (0.806)** — highest generalization
- ✅ Matches base on 7/10 questions
- ✅ Long-term debt: gap goes from +0.60 (p_gpt54) → **−0.08** (v2 actually generalizes *better* than base on this Q)
- ❌ Fails to match base on 3 questions: net income (0.80/0.90), total assets (0.80/1.00), trading symbols (0.90/1.00)
- ❌ Trading-symbols bloated to 26 rules (hit `max_extra_rules=20` cap without converging)

### Why p_v2 doesn't fully match base

| Mechanism | Cause | Affected Qs |
|-----------|-------|-------------|
| Phase B cap | `max_extra_rules=20` hit before reaching base | trading symbols |
| Cost-effectiveness order skips key rules | The rule v1 needed to hit base is deep in cost-effectiveness order, never reached | total assets |
| Phase C undoes Phase B | Phase B adds N rules to nominally hit base, Phase C prunes most as redundant — net effect "1-2 useful additions" | net income |
| LLM stochasticity | base measured weeks ago; current eval ~0.04 noise floor | all |

---

## p_v3 — v2 with cumulative-prefix fallback (Phase B′)

**Algorithm:** identical to p_v2 except Phase B has a fallback mode:
1. Phase A as in v2.
2. **Phase B (single-rule, primary):** add the next rule by cost-effectiveness order; eval. Up to `max_extra_rules_single=20` additions.
3. **Phase B′ (cumulative-prefix, fallback, new):** if Phase B hit the cap without `merge_acc ≥ base`, switch to v1-style exponential-prefix testing on the *full remaining pool*. For k=1,2,4,8,…|remaining|, test `S ∪ remaining[:k]`. Once a k satisfies `merge_acc ≥ base`, commit and exit.
4. Phase C backward pruning (same as v2).

**Why this should match base:** Phase B′ uses cumulative prefixes — by k=|pool|, we're testing the entire rule set, which by definition gives `base` accuracy (modulo LLM noise). So unless `base` itself is unreachable in this run (LLM stochasticity), Phase B′ converges.

**Expected trade-off vs v2:**
- More rules in S after Phase B′ (potentially many for hard questions)
- Phase C then trims redundancy
- Final rule count may grow on hard questions (esp. total assets, trading symbols)
- uAcc may dip slightly (more rules → more retrieval interference on unsampled)
- sAcc should match base on all 10 questions

**Hyperparameters:**
- `max_extra_rules_single` = 20 (same as v2)
- `max_prefix_k_iters` = 8 (caps exp-search depth: k can hit 1,2,4,8,…128, plus a final |remaining| try)
- All other hyperparams inherited from v2

---

## How to add a new variant

1. Create `src/rule_refinement/select_rules_pareto_<name>.py` — new selection module.
2. Create `test/run_select_all_pareto_<name>.py` — driver (clone from existing, change output paths and `run_selection_pareto_*` import).
3. Create `test/eval_pareto_<name>_sampled.py` and `test/eval_pareto_<name>_unsampled.py` — eval drivers pointing at the new selection folder.
4. Output folders:
   - `results/.../selected_rules_pareto_<name>/`
   - `results/.../selector_run_pareto_<name>/` (intermediate merge logs)
   - `results/.../eval_pareto_<name>/`
5. Add a row to the summary table above.
6. Add a section describing the algorithm, motivation, and key findings.

---

## Code locations

| File | Role |
|------|------|
| `src/rule_refinement/select_rules.py` | `_greedy_cover` — reused by every Pareto variant |
| `src/rule_refinement/select_rules_pareto.py` | p_mini and p_gpt54 (same module, different driver `MODEL_NAME`) — also exports utility functions reused by v2/v3 |
| `src/rule_refinement/select_rules_pareto_proxy.py` | p_proxy |
| `src/rule_refinement/select_rules_pareto_v2.py` | p_v2 — adds Phase B + C |
| `src/rule_refinement/select_rules_pareto_v3.py` | p_v3 — adds Phase B′ fallback to v2 |
| `src/rule_refinement/eval_judge.py` | `judge` (LLM) + `proxy_judge` (substring) |
| `src/rule_refinement/cost_profile.py` | Phase 0 cost profile (shared cache across all variants) |
| `src/rule_refinement/baseline_targets.py` | `load_target_docs(eval_merge_path)` → D\* |
| `src/rule_refinement/coverage_check.py` | `load_or_compute_coverage` for cov_map |

Drivers in `test/`:
- `run_select_all_pareto.py` → p_mini
- `run_select_all_pareto_proxy.py` → p_proxy
- `run_select_all_pareto_gpt54.py` → p_gpt54
- `run_select_all_pareto_v2.py` → p_v2
- `run_select_all_pareto_v3.py` → p_v3
- `eval_pareto_{name}_{sampled,unsampled}.py` for each variant
