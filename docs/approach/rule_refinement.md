# Rule Refinement and Selection

This document describes every rule refinement and selection approach in the LSF codebase. Each approach takes a generated rule pool (see `docs/rule_generation.md`) and produces a smaller working set that is applied at inference time. The goal is to maintain accuracy on sampled docs while generalizing well to unseen docs at lower retrieval cost.

**Evaluation dataset:** FinanceBench. `sAcc` = merge accuracy on sampled docs, `uAcc` = merge accuracy on unsampled docs, `cost` = mean(retrieved\_tokens / total\_doc\_tokens). Rule pool source unless noted: LLM-coarse gpt54 (~63 rules/question on single-cluster, ~35 on multi-cluster).

---

## Results summary — single cluster (10 sampled, 50 unsampled, 10 questions)

| Variant | Code | sAcc | uAcc | cost\_s | cost\_u | Mean rules | Overfit gap |
|---------|------|-----:|-----:|--------:|--------:|-----------:|------------:|
| **Base (full pool)** | — | 0.910 | **0.892** | 0.172 | 0.169 | ~63 | +0.018 |
| **v1** | `src/rule_refine/v1.py` | 0.890 | 0.778 | 0.029 | 0.035 | 7.8 | +0.112 |
| **p\_mini** | `src/rule_refine/selection/select_rules_pareto.py` (gpt54mini) | 0.840 | 0.780 | 0.005 | 0.005 | 3.9 | +0.060 |
| **p\_proxy** | `src/rule_refine/selection/select_rules_pareto_proxy.py` | 0.843\* | 0.737\* | 0.004 | 0.004 | 3.0 | +0.106 |
| **p\_gpt54** | `src/rule_refine/selection/select_rules_pareto.py` (gpt54) | 0.880 | 0.748 | 0.007 | 0.005 | 4.3 | +0.132 |
| **p\_v2** | `src/rule_refine/selection/select_rules_pareto_v2.py` | 0.880 | 0.806 | 0.012 | 0.010 | 5.1 | +0.074 |
| **p\_v3** | `src/rule_refine/selection/select_rules_pareto_v3.py` | **0.910** | 0.804 | 0.011 | 0.009 | 4.7 | +0.106 |
| **agentic** | `src/rule_refine/agentic.py` | **0.940** ⭐ | 0.870 | 0.027 | 0.023 | **2.0** ⭐ | +0.070 |
| **agentic_codex** | `src/rule_refine/agentic_codex.py` | — | — | — | — | — | — |
| **fallback** | `src/rule_apply/default.py` | — | **0.892** | — | 0.030 | 5.1+on-demand | — |

\* p_proxy on 7/10 questions only — fails completely on 3 where GT strings are not verbatim substrings (state/EIN, total revenue, trading symbols).

**Key takeaways:**
- **fallback** matches base uAcc (0.892) at 18% of base retrieval cost — Pareto-dominant.
- **agentic** achieves the highest sAcc (0.940, even above base), second-best uAcc (0.870), and fewest rules (2.0/Q). The "less-is-more" effect: focused retrieval is more reliably answered by gpt54 than noisy full-pool unions.
- **p_v2** is the best selection-only algorithm (uAcc 0.806) without agentic infrastructure.
- **p_mini** (cheap judge) generalizes better than **p_gpt54** (strong judge) — counter-intuitive: the noisy gpt54mini verdicts act as accidental regularization against overfit specialists.
- Matching base sAcc costs uAcc: p_v3 forces sAcc=0.910 but pays 2× LLM budget with no uAcc gain over p_v2.

---

## Results summary — multi cluster (18 sampled, 68 unsampled, 12 questions)

Strategy run: agentic selection + fallback deployment.

| Variant | Phase | sAcc | uAcc | cost\_s | cost\_u | Mean rules |
|---------|-------|-----:|-----:|--------:|--------:|-----------:|
| Base (full pool) | — | **0.972** | 0.935 | 0.036 | 0.059 | ~35 |
| Agentic + fallback | selection + deploy | — | **0.940** | — | **0.009** | 2.7+on-demand |

Agentic+fallback marginally beats the full pool on uAcc (0.940 vs 0.935) at 6.5× lower retrieval cost. Mean fallback rate 11.2%.

---

## Approach 1 — v1 (cost-sort + exponential search + backward prune)

**Code:** `src/rule_refine/v1.py`  
**Doc:** `docs/rule_refine.md`

### Description

Given a full rule pool, select a minimal-cost subset whose merge accuracy matches the target (the full pool's accuracy on sampled docs). No LLM calls needed for cost computation; accuracy is measured by LLM QA + judge on the union of retrieved spans.

### Interface

```python
def rule_refine(
    rule_names: list[str],
    target_accuracy: float,
    question: str,
    question_slug: str,
    documents: list[dict],
    ground_truth: dict,
    rules_dir: str = "rules/llm/financebench",
    output_dir: str = "results/llm_rule_refine",
    model_name: str = "gpt54",
) -> dict
```

### Algorithm

1. Sort all rules by `avg_cost_ratio` ascending (cheapest first).
2. **Exponential search:** try k=1,2,4,8,… cheapest rules until `merge_acc ≥ target_accuracy`.
3. **Backward prune:** remove any rule whose removal still preserves `merge_acc ≥ target`.
4. In-loop judge: gpt54.

### Results (FinanceBench, single cluster, gpt54)

| sAcc | uAcc | cost\_s | cost\_u | Mean rules | LLM calls/Q |
|-----:|-----:|--------:|--------:|-----------:|------------:|
| 0.890 | 0.778 | 0.029 | 0.035 | 7.8 | ~864 |

**Dominant failure:** cost-sort structural exclusion — high-coverage, high-cost rules are never considered because the exponential search terminates on cheap rules first. Worst overfit gap (+0.112) of selection-only variants. Dominated by p_v2 (same uAcc, more rules, more LLM cost).

---

## Approach 2 — Pareto variants

**Code:** `src/rule_refine/selection/` (multiple files)  
**Doc:** `docs/pareto_versions.md`, `docs/rule_selection_pareto_implementation.md`

All Pareto variants share the same core primitive: **cost-effectiveness greedy cover** — sort rules by `cov(r) / avg_cost_ratio(r)` descending, then greedily admit rules that cover at least one uncovered doc under the in-loop judge.

### p\_mini — Pareto with gpt54mini judge ⭐ Recommended

**Code:** `src/rule_refine/selection/select_rules_pareto.py` (with `MODEL_NAME=gpt54mini`)  
**Driver:** `test/run_select_all_pareto.py`

In-loop judge: gpt54mini. The cheap judge's noisier verdicts act as accidental regularization — marginal specialists get rejected, broader rules get admitted. Lowest overfit gap (+0.060) of all judge-aware variants.

| sAcc | uAcc | cost\_u | Mean rules | Wall time |
|-----:|-----:|--------:|-----------:|----------:|
| 0.840 | 0.780 | 0.005 | 3.9 | ~37 min |

---

### p\_proxy — Pareto with substring-only judge (zero LLM)

**Code:** `src/rule_refine/selection/select_rules_pareto_proxy.py`  
**Driver:** `test/run_select_all_pareto_proxy.py`

In-loop judge: case-insensitive substring check `gt.lower() in retrieved_text.lower()`. Zero LLM calls during selection.

| sAcc\* | uAcc\* | cost\_u | Mean rules | Wall time |
|-------:|-------:|--------:|-----------:|----------:|
| 0.843 | 0.737 | 0.004 | 3.0 | ~10 min |

\* On 7/10 questions only. Fails completely on 3 where GT strings don't appear verbatim (e.g. "Delaware" vs "DE", "$12,345 million" vs "12,345"). On all 10 with failures counted as 0: sAcc=0.590, uAcc=0.516.

**Recommended use:** fast pre-filter or debugging. Not deployable as-is without a fuzzy numeric fallback.

---

### p\_gpt54 — Pareto with gpt54 judge

**Code:** `src/rule_refine/selection/select_rules_pareto.py` (with `MODEL_NAME=gpt54`)  
**Driver:** `test/run_select_all_pareto_gpt54.py`

Same algorithm as p_mini, stronger judge. Counter-intuitively worse generalization (uAcc 0.748 vs p_mini 0.780). The gpt54 judge confidently admits narrow specialists that overfit; p_mini's noisy judge rejects them.

| sAcc | uAcc | cost\_u | Mean rules |
|-----:|-----:|--------:|-----------:|
| 0.880 | 0.748 | 0.005 | 4.3 |

---

### p\_v2 — Pareto with accuracy floor + backward prune

**Code:** `src/rule_refine/selection/select_rules_pareto_v2.py`  
**Driver:** `test/run_select_all_pareto_v2.py`

Grafts v1's accuracy guarantee onto Pareto's cost-effectiveness ordering.

**Algorithm:**
1. **Phase A:** cost-effectiveness greedy cover (same as p_gpt54).
2. **Phase B (accuracy floor):** if `merge_acc(S) < base`, admit next rule by cost-effectiveness order; re-eval. Repeat up to `max_extra_rules=20`.
3. **Phase C (backward prune):** drop any rule whose removal preserves `merge_acc ≥ base`.

| sAcc | uAcc | cost\_u | Mean rules | Wall time |
|-----:|-----:|--------:|-----------:|----------:|
| 0.880 | **0.806** | 0.010 | 5.1 | ~66 min |

**Highest uAcc of any selection-only variant.** Matches base on 7/10 questions; misses on 3 (net income, total assets, trading symbols) due to Phase B cap or cost-effectiveness ordering.

---

### p\_v3 — v2 with cumulative-prefix Phase B′

**Code:** `src/rule_refine/selection/select_rules_pareto_v3.py`  
**Driver:** `test/run_select_all_pareto_v3.py`

Adds a fallback to p_v2 Phase B: if the single-rule addition loop hits `max_extra_rules` without reaching base, switch to v1-style exponential prefix search on the remaining pool.

**Algorithm:** same as p_v2 except after Phase B exhaustion → Phase B′: test `S ∪ remaining[:k]` for k=1,2,4,8,… until `merge_acc ≥ base` or pool exhausted → Phase C backward prune.

| sAcc | uAcc | cost\_u | Mean rules | LLM calls vs p\_v2 |
|-----:|-----:|--------:|-----------:|-------------------:|
| **0.910** | 0.804 | 0.009 | 4.7 | ~2× |

Matches base sAcc on 8/10 questions. uAcc essentially tied with p_v2 (0.804 vs 0.806). Costs 2× the LLM budget for +0.030 sAcc and no uAcc gain — not worth it unless matching base is a hard requirement.

---

## Approach 3 — Agentic selection (Claude Opus 4.7) ⭐ Recommended

**Code:** `src/rule_refine/agentic.py` (driver), `src/rule_refine/agentic_task_prompt.md` (prompt)  
**Doc:** `docs/rule_selection_agentic.md`

### Description

Claude Opus 4.7 (`claude -p`) acts as the outer-loop optimizer. The agent inspects rules semantically (reads docstrings, spots overfit-prone rules), calls tool-wrapped primitives to measure cost/coverage/accuracy, proposes subsets, and iterates until the hard accuracy constraint is met or budget is exhausted.

One session per question. The agent can allocate more verification calls to hard questions and fewer to easy ones — algorithmic variants cannot.

### Hard and soft constraints

| Type | Constraint | Tool |
|------|------------|------|
| Hard | `A(S, d) = 1` for every `d ∈ D*_s` | `verify_accuracy` (~20 gpt54 calls each) |
| Soft | Minimize `Σ avg_cost_ratio(r)` | `compute_cost` (free) |
| Soft | Maximize `min cov(r)` | `compute_coverage` (free if cached) |
| Soft | Keep `|S|` small | Agent working memory |

Budget: 30 `verify_accuracy` calls per question. The agent produces a natural-language `rationale` per rule pick.

### Interface (driver)

```bash
python src/rule_refine/agentic.py --sample-set random [--slug <slug>] [--dry-run]
python src/rule_refine/agentic.py --sample-set multi
```

### Results (FinanceBench, single cluster, opus47)

| sAcc | uAcc | cost\_u | Mean rules | Avg latency | Approx cost/Q |
|-----:|-----:|--------:|-----------:|------------:|--------------:|
| **0.940** ⭐ | 0.870 | 0.023 | **2.0** ⭐ | 186s | ~$1.16 |

Highest sAcc of any variant — exceeds even the full pool base (0.910). The "less-is-more" effect: 2-rule selections produce focused retrieval that gpt54 extracts from more reliably than the full-pool union. 7/10 questions at sAcc=1.00; 3 at sAcc=0.90; long-term debt at sAcc=0.70 (structurally hard).

**Why agentic beats algorithmic variants:**
- Reads docstrings semantically (e.g. avoids `rule_*_page_around_39` as overfit-prone)
- Allocates verify budget per-question based on difficulty
- Produces auditable `rationale` field per selected rule
- Trades off cost vs accuracy adaptively

**Output:**
- `results/.../selected_rules_agent/<slug>.json`
- `results/.../agent_trace/<slug>.jsonl`
- `results/.../eval_agentic/<slug>_{sampled,unsampled}.json`

---

## Approach 3b — Agentic selection (Codex / gpt-5.4)

**Code:** `src/rule_refine/agentic_codex.py` (driver), `src/rule_refine/agentic_task_prompt.md` (prompt, shared with Approach 3)

### Description

Direct Codex equivalent of Approach 3. Uses exactly the same task prompt, the
same hints, the same constraints, the same output contract — only the
underlying CLI is swapped from `claude -p` to `codex exec`, and the agent
backbone is gpt-5.4 (or gpt-5.4-mini) instead of Claude Opus 4.7. The driver
spawns one Codex agent session per question; there is no Claude Code wrapper.

The intent is to isolate the effect of the agent backbone (Claude Opus vs.
Codex gpt-5.4 / gpt-5.4-mini) while keeping the prompt, tool semantics, and
objectives identical to Approach 3.

### Models

- `gpt54` — `gpt-5.4` via Codex CLI
- `gpt54mini` — `gpt-5.4-mini` via Codex CLI

### Invocation

```bash
# Set the Azure key once per shell (see docs/codex_setup.md for the YAML extraction)
export AZURE_OPENAI_API_KEY=$(awk -F': ' '/^api_key:/{print $2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt)

# All questions:
python src/rule_refine/agentic_codex.py --model gpt54

# Single question:
python src/rule_refine/agentic_codex.py --slug what_is_the_registrants_telephone_number_10_llm

# Lower verify budget:
python src/rule_refine/agentic_codex.py --budget 20
```

### Status

Implemented. No benchmark results yet — runs alongside Approach 3 in the
pipeline test grid (`docs/pipeline.md`).

---

## Approach 4 — Fallback deployment (moved)

The fallback strategy is a deployment-time rule-application strategy, not a refinement algorithm. It's now documented in [rule_application.md](./rule_application.md) as **Strategy 3 — Default (refined-with-fallback)**. Code: `src/rule_apply/default.py`.

When paired with a refinement variant from this doc (typically p_v2) it recovers the refined→base generalization gap at a small extra LLM cost. See rule_application.md for details and results.

---

## Approach 5 — Static-τ and auto-tighten (spec only)

**Code:** `src/rule_refine/selection/select_rules.py`, `src/rule_refine/selection/select_rules_auto_tighten.py`  
**Not benchmarked.**

These are earlier algorithmic variants implemented in code but without measured results:
- **static-τ:** apply a fixed cost threshold; keep all rules below τ.
- **auto-tighten:** iteratively lower τ until accuracy would drop, then stop.

Both are superseded by the Pareto variants and are retained for reference.

---

## Approach 6 — v2 refinement proposal (spec only)

**Code:** `src/rule_refine/v2/`  
**Doc:** `docs/rule_refine_v2.md`  
**Not implemented.**

A redesign of v1 that addresses three root-cause failure modes identified in the overfit analysis:

| Failure mode | Share | Cause |
|---|--:|---|
| Sampling blind spots | ~40% | Rules with zero sampled coverage eliminated even when they generalize well |
| Small-N pruning noise | ~35% | Rules helping only 1–2 sampled docs pruned despite being critical on unsampled |
| Rule specificity | ~25% | Page/date/format-specific rules match sampled by coincidence |

The proposal replaces cost-sort + exponential search with a coverage-aware selection that explicitly considers unsampled generalization signals during selection. Spec complete; implementation pending.

---

## Approach comparison

| Approach | Code exists | Selection oracle | LLM calls | Best sAcc | Best uAcc | Cost |
|----------|-------------|-----------------|-----------|----------:|----------:|-----:|
| v1 | Yes | gpt54 | ~864/Q | 0.890 | 0.778 | 0.035 |
| p_mini | Yes | gpt54mini | ~low | 0.840 | 0.780 | 0.005 |
| p_proxy | Yes | substring (0 LLM) | 0 | 0.843\* | 0.737\* | 0.004 |
| p_gpt54 | Yes | gpt54 | ~medium | 0.880 | 0.748 | 0.005 |
| p_v2 | Yes | gpt54 | ~medium | 0.880 | **0.806** | 0.010 |
| p_v3 | Yes | gpt54 | ~2× p_v2 | **0.910** | 0.804 | 0.009 |
| agentic | Yes | Claude+gpt54 | ~30 verify | **0.940** | 0.870 | 0.023 |
| fallback | Yes | gpt54mini gate | per-doc | — | **0.892** | 0.030 |
| static-τ | Yes (no results) | — | — | — | — | — |
| auto-tighten | Yes (no results) | — | — | — | — | — |
| v2 proposal | Spec only | — | — | — | — | — |

---

## File index

### Selection algorithms
| File | Variant |
|------|---------|
| `src/rule_refine/v1.py` | v1 |
| `src/rule_refine/selection/select_rules.py` | static-τ |
| `src/rule_refine/selection/select_rules_auto_tighten.py` | auto-tighten |
| `src/rule_refine/selection/select_rules_pareto.py` | p_mini + p_gpt54 |
| `src/rule_refine/selection/select_rules_pareto_proxy.py` | p_proxy |
| `src/rule_refine/selection/select_rules_pareto_v2.py` | p_v2 |
| `src/rule_refine/selection/select_rules_pareto_v3.py` | p_v3 |
| `src/rule_refine/agentic.py` | agentic |
| `src/rule_apply/default.py` | fallback |

### Shared primitives
| File | Purpose |
|------|---------|
| `src/rule_refine/selection/eval_judge.py` | `judge` (LLM) + `proxy_judge` (substring) |
| `src/rule_refine/selection/cost_profile.py` | Cost cache (no LLM) |
| `src/rule_refine/selection/baseline_targets.py` | Loads D\* (docs answerable by full pool) |
| `src/rule_refine/selection/coverage_check.py` | Per-rule coverage map |
| `src/rule_apply/merge.py` | Apply any rule set to a doc (union retrieval) |

### Output directories (under `results/financebench/lsf/single_cluster/llm/gpt54/one_shot/`)
| Folder | Contents |
|--------|----------|
| `selected_rules_pareto_gpt54mini/` | p_mini selections |
| `selected_rules_pareto_proxy/` | p_proxy selections |
| `selected_rules_pareto_gpt54/` | p_gpt54 selections |
| `selected_rules_pareto_v2/` | p_v2 selections |
| `selected_rules_pareto_v3/` | p_v3 selections |
| `selected_rules_agent/` | Agentic selections |
| `eval_pareto_*/` | Per-variant eval JSONs |
| `eval_pareto_v2_fallback/` | Fallback eval JSONs |
| `eval_agentic/` | Agentic eval JSONs |
| `eval_merge/` | Full-pool baseline |
| `cost_profile/` | Shared cost cache |
| `eval_individual/` | Per-rule coverage |
