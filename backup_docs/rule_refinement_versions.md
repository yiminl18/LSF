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
| **agentic** | `agent/run_agent_select.py` | selection | **0.940** ⭐ | 0.0272 | **0.870** | 0.0230 | **2.0** ⭐ | +0.070 | Claude Opus 4.7 outer loop with tool calls — highest sAcc of any variant, even above base |
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

**Result (run completed 2026-05-18 21:53 UTC, 53 min wallclock):**
- **Mean sAcc: 0.940 ⭐** — highest of any variant, exceeds even `base` (0.910)
- **Mean uAcc: 0.870** — second-best (only fallback beats it at 0.892)
- **Mean rules selected: 2.0 ⭐** — smallest of any variant (vs p_v2's 5.1, v1's 7.8)
- Cost: ~$1.16 per question (Opus reasoning + gpt54 verification)
- 7/10 questions at sAcc = 1.00 perfect; 3 at sAcc = 0.90; 1 at sAcc = 0.70 (long-term debt, the structurally hard one)

**Token usage & cost (selection process, per question, averaged over 10 questions):**
- Avg latency: **186.2s**
- Avg gpt54 verify tokens (input + output): **61,844** → cost ratio **0.70** (vs avg single-doc tokens 88,432)
- Opus 4.7 reasoning tokens: **not logged** — only the gpt54 verification tool calls are recorded in `selected_rules_agent/<slug>.json` (`tool_input_tokens`, `tool_output_tokens`); the Opus outer-loop tokens are not captured
- The 0.70 cost ratio covers the verification side only; total cost including Opus reasoning accounts for the ~$1.16/Q figure above

The agent's "less-is-more" effect: small, clean rule sets retrieve focused text that gpt54 extracts from more reliably than the full pool's union.

**Why agentic beats algorithmic variants:**
- Reads docstrings semantically (e.g. avoids `rule_*_page_around_39` as overfit-prone)
- Decides budget allocation per question (more verifications on hard questions)
- Produces a natural-language `rationale` per pick — auditable
- Trades off cost vs accuracy adaptively

**Specs:** `docs/rule_selection_agentic.md`, `agent/task_prompt.md`.
**Outputs:** `results/.../selected_rules_agent/<slug>.json`, `agent_trace/<slug>.jsonl`, `eval_agentic/<slug>_{sampled,unsampled}.json`.

---

### fallback (deployment-time, not selection) — `src/default_rule.py`

**Rule source:** both the refined subset and the full-pool fallback are drawn from the **LLM-coarse (gpt54, one-shot)** rule pool (`rules/financebench/lsf/single_cluster/llm/gpt54/one_shot/<slug>_10_llm/`). The refined subset is specifically the **p_v2** selection over that pool (`selected_rules_pareto_v2/<slug>.json`), averaging 5.1 rules per question out of the ~63-rule pool.

**Algorithm:** at inference on each unsampled doc:
1. Apply **p_v2 refined rules** → retrieved text
2. Ask **gpt54mini** "does this text contain the answer?"
3. If YES → answer with gpt54 on refined retrieval
4. If NO → fall back to **full LLM-coarse pool** (~63 rules), then answer with gpt54

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

## Multi-cluster results

Dataset: **18 sampled docs, 68 unsampled docs, 12 questions** (10-K, 10-Q, 8-K filings mixed).
Rule pool: `rules/financebench/lsf/multi_clusters/llm/gpt54/one_shot/`, `_18_llm` slug suffix.
Strategy run: **agentic selection** (Claude Opus 4.7) → **fallback deployment** (gpt54mini gate + full-pool fallback).

> Note: eval_merge/perf_summary.json covers 10 of the 12 questions (long-term debt and exhibit listing
> were added after the initial sampled eval). The agentic+fallback summary covers all 12.

### Summary table

| Variant | Phase | sAcc (18) | cost_ratio_s | uAcc (68) | cost_ratio_u | Mean rules |
|---------|-------|----------:|-------------:|----------:|-------------:|-----------:|
| **Full pool (base)** | — | **0.972** | 0.0356 | 0.935 | 0.0589 | ~35 avg |
| **Agentic+fallback** | selection+deploy | — | — | **0.940** | **0.0090** | 2.7 + on-demand |

Cost ratio = gpt54 input tokens retrieved per doc / total doc tokens (lower = cheaper).
Agentic+fallback cost ratio uses `gpt54_in_per_doc / avg_doc_tokens` (avg_doc_tokens ≈ 47,819).

### Per-question breakdown (10 questions with full-pool baseline)

| Question | sAcc (full pool) | uAcc (full pool) | uAcc (agentic+fb) | fbRate | cost_ratio_s | cost_ratio_u | cost_ratio_fb |
|----------|----------------:|----------------:|------------------:|-------:|-------------:|-------------:|--------------:|
| Office address (city/state) | 0.944 | 1.000 | **1.000** | 5.9% | 0.0198 | 0.0212 | 0.0083 |
| Doc type (form) | 1.000 | 1.000 | 0.985 | 2.9% | 0.1294 | 0.3754 | **0.0070** |
| Phone number | 1.000 | 0.956 | 0.971 | 4.4% | 0.0085 | 0.0071 | 0.0064 |
| State / IRS EIN | 1.000 | 0.971 | 0.971 | 5.9% | 0.0890 | 0.0778 | 0.0094 |
| Stock exchange | 1.000 | 0.956 | 0.956 | 14.7% | 0.0030 | 0.0059 | 0.0069 |
| Reporting period | 0.889 | 0.941 | **0.956** | 5.9% | 0.0079 | 0.0113 | 0.0086 |
| Registrant name | 1.000 | 0.941 | 0.941 | 10.3% | 0.0230 | 0.0181 | 0.0065 |
| Office address (ZIP) | 0.944 | 0.912 | **0.941** | 20.6% | 0.0125 | 0.0150 | 0.0084 |
| Company name (exact) | 1.000 | 0.912 | 0.912 | 14.7% | 0.0059 | 0.0039 | 0.0065 |
| Trading symbols | 0.944 | 0.765 | 0.765 | 26.5% | 0.0571 | 0.0529 | 0.0221 |
| **MEAN** | **0.972** | **0.935** | **0.940** | **11.2%** | **0.0356** | **0.0589** | **0.0090** |

Additional 2 questions (no full-pool baseline):

| Question | uAcc (agentic+fb) | fbRate |
|----------|------------------:|-------:|
| Long-term debt | 0.500 | 16.2% |
| Exhibit listing | 0.279 | 20.6% |

### Key findings

1. **Agentic+fallback marginally beats the full pool on unsampled (0.940 vs 0.935)** while using only 2.7 selected rules per question (vs ~35 in the full pool).

2. **Cost ratio drops 6.5× vs full pool unsampled (0.0090 vs 0.0589)**. Most dramatic case: "Doc type (form)" cost_ratio falls from 0.375 → 0.007 — the agentic agent correctly identified that the full pool was retrieving large, redundant spans for this question.

3. **Mean fallback rate 11.2%** — the gpt54mini gate triggers on ~7 of 68 unsampled docs per question, recovering from rule misses at modest cost.

4. **Trading symbols and exhibit listing remain hard** (uAcc 0.765 and 0.279). Trading symbols suffer from format diversity across 10-K/10-Q/8-K forms; exhibit listing requires free-form extraction across heterogeneous indices.

5. **Long-term debt at 0.500** — structurally hard across both single-cluster (base 0.66) and multi-cluster; the question requires numeric extraction across varied table formats.

### Outputs

All under `results/financebench/lsf/multi_clusters/llm/gpt54/one_shot/`:

| Folder | Contents |
|--------|----------|
| `selected_rules_agent/` | 12 per-question agentic selection JSONs (`_18_llm` slugs) |
| `agent_trace/` | 12 JSONL step-by-step agent traces |
| `eval_agentic_fallback/` | 12 per-question unsampled eval JSONs + `summary.json` |
| `eval_merge/` | Full-pool baseline (`perf_summary.json`, 10 questions) |

Plot: `analysis/multi_cluster_results.png` — accuracy + cost ratio + $/doc comparison.

---

## Aggregate findings

### Which selection variant has the best sAcc?
**agentic** — sAcc 0.940 ⭐, **exceeds even base (0.910)**. Only 2.0 rules per question on average. Cost ~$1.16/Q.

### Which selection variant generalizes best (uAcc)?
- **fallback** at uAcc 0.892 (matches base exactly) — deployment-time strategy.
- **agentic** at uAcc 0.870 — selection-time, second-best uAcc.
- **p_v2** at uAcc 0.806 — best selection-only variant if you can't afford agentic.

### Which is most cost-efficient?
For **accuracy-matching the full pool**: fallback (uAcc 0.892 at cost 0.030 = **5.6× cheaper than base**, 3× costlier than refined-only).

For **selection alone, cheap**: p_proxy (~$0.005/run) when GTs are verbatim strings; p_mini when they aren't.

For **selection alone, accuracy-first**: agentic (~$1.16/Q) — most expensive but highest accuracy and smallest rule set.

### What doesn't work
- p_proxy on numeric/string-format answers (3/10 Qs fail).
- v1's exhaustive search is dominated by p_v2 (same uAcc, more rules, more cost).
- p_gpt54 is dominated by p_mini (same algorithm, gpt54 judge produces worse uAcc).

### Surprising findings
1. **Agentic beats base on sAcc.** Agent's 2-rule selections produce cleaner retrieval than the full pool's union. The "less-is-more" effect — gpt54 extracts from focused text more reliably than noisy union.
2. **Cheaper in-loop judge → better generalization** (p_mini vs p_gpt54). gpt54mini's noisier verdicts regularize against over-confident specialist admission.
3. **Fallback can beat base on individual questions** (long-term debt 0.72 vs base 0.66). The gate filters out retrievals that would confuse gpt54.
4. **Matching base sAcc costs uAcc on algorithmic variants** — p_v3 forces sAcc=0.91 but uAcc stays at 0.80, paying 2× LLM budget for nothing. Agentic is the exception — it matches+exceeds base sAcc without uAcc regression.

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

### Output directories (all under `results/financebench/lsf/single_cluster/llm/gpt54/one_shot/`)
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
| Multi | 18 | 68 | 12 | ~35 avg (agentic+fallback run completed 2026-05-21) |
