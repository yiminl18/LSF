# Rule Generation — Version Tracking

This document indexes every rule-generation strategy implemented in the LSF codebase, with the merge-accuracy and cost numbers measured on FinanceBench. Each generation method produces a rule pool that downstream refinement / selection algorithms (see `docs/rule_refinement_versions.md`) then filter or trim.

Companion docs:
- `docs/rule_gen_llm_coarse.md` — LLM-coarse spec
- `docs/rule_gen_agent.md`, `rule_gen_agent_coarse.md`, `rule_gen_agent_exact.md` — agent variants
- `docs/agent.md` — Claude Code agent execution conventions

---

## Summary table — accuracy and cost of the *full rule pool* (no refinement)

All numbers come from `eval_merge/<slug>_{sampled,unsampled}.json` produced by `test/run_eval_merge_sampled.py` and `test/run_eval_merge_unsampled.py`. Accuracy uses gpt54 QA + gpt54 judge; `cost_ratio` is mean over docs of `retrieved_tokens / total_doc_tokens`.

| Method | Module | Cluster | Gen model | N(Q sampled) | sAcc | cost_s | N(Q unsampled) | uAcc | cost_u |
|--------|--------|---------|-----------|-------------:|-----:|-------:|---------------:|-----:|-------:|
| **LLM-coarse** | `src/rule_gen_llm_coarse.py` | single | gpt54 | 10 | **0.910** | 0.1722 | 10 | **0.892** | 0.1686 |
| **Agent-raw** | `src/rule_gen_agent_claude.py` | single | gpt54 | 10 | 0.860 | 0.1155 | 10 | 0.754 | 0.1060 |
| **Agent-raw** | `src/rule_gen_agent_claude.py` | single | opus47 | 9 | 0.889 | 0.1312 | 9 | 0.776 | 0.1198 |
| **Agent-refined** | (agent post-process) | single | gpt54 | 10 | 0.880 | **0.0363** | 10 | 0.734 | **0.0364** |
| **Agent-raw** | `src/rule_gen_agent_claude.py` | **multi** | gpt54 | 12 | 0.856 | 0.0679 | 12 | **0.843** | 0.0869 |
| **Agent-raw** | `src/rule_gen_agent_claude.py` | **multi** | opus47 | 12 | 0.852 | **0.0127** | 12 | 0.705 | **0.0125** |
| LLM-coarse | `src/rule_gen_llm_coarse.py` | multi | gpt54 | — | — | — | — | — | — (not yet evaluated) |

> Dataset: FinanceBench. "single cluster" = sample drawn from 60 FinanceBench docs (10 sampled, 50 unsampled). "multi cluster" = 18 sampled, 96 unsampled (12 questions instead of 10).

---

## Method descriptions

### 1. LLM-coarse (`src/rule_gen_llm_coarse.py`)
- **Approach**: single prompt to gpt54 with the question + a few example documents. The model emits a Python span-retrieval function in one shot.
- **Result on single cluster**: 100 rules generated per question on average. Full-pool merge: **sAcc=0.91, uAcc=0.89**. Highest accuracy on both splits — but also highest retrieval cost (~17% of doc tokens). The "default" rule pool everything else is compared against.
- **Output dir**: `rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm/`
- **Spec**: `docs/rule_gen_llm_coarse.md`

### 2. Agent-raw (`src/rule_gen_agent_claude.py` / `src/rule_gen_agent.py`)
- **Approach**: Claude Code (`claude -p`) opens an interactive session with full tool access (file read/write, Python exec). Iteratively builds rules, evaluates them, refines until target merge accuracy or budget exhaustion.
- **Result on single cluster (gpt54-generated)**: sAcc=0.86, uAcc=0.75. Lower accuracy than LLM-coarse but **~35% cheaper retrieval** (cost_s=0.115 vs 0.172). The agent compresses rules during generation.
- **Result on single cluster (opus47-generated)**: sAcc=0.89, uAcc=0.78. Better than gpt54-generated agent rules on both axes, slightly worse on cost (0.13 vs 0.12).
- **Output dirs**:
  - `rules/financebench_single_cluster/agent/gpt54/raw/<slug>_10_agent/`
  - `rules/financebench_single_cluster/agent/opus47/raw/<slug>_10_agent/`
- **Spec**: `docs/rule_gen_agent.md`

### 3. Agent-refined (post-process pass on Agent-raw)
- **Approach**: a second agent pass takes Agent-raw rules and merges/specializes them to reduce retrieval cost while preserving sampled accuracy.
- **Result on single cluster (gpt54)**: sAcc=0.88 (small drop from 0.86 base — actually slight improvement), uAcc=**0.734** (regression from raw 0.754). cost=**0.036** (3× cheaper than raw).
- **Trade-off**: large cost win on sampled (essentially matches sampled-refined cost ratios from algorithmic refinement); modest uAcc regression. Overfitting risk: refinement on the sampled-only signal compresses too aggressively.
- **Output dir**: `rules/financebench_single_cluster/agent/gpt54/refined/<slug>_10_agent_refined/`

### 4. Multi-cluster Agent-raw
- **Approach**: same agent rule-generation pipeline applied to the 18-doc multi-cluster sample set (covers 10-K, 10-Q, 8-K, earnings releases).
- **Result on multi-cluster (gpt54)**: sAcc=0.86, **uAcc=0.84** (much smaller sampled→unsampled gap than single-cluster's 0.86→0.75). Diverse sampled docs apparently produce more generalizable rules.
- **Result on multi-cluster (opus47)**: sAcc=0.85, uAcc=0.71, cost=**0.013** (10× cheaper than gpt54-multi). Opus rules are compact but less robust on unsampled.
- **Output dirs**: `rules/financebench_multi_clusters/agent/{gpt54,opus47}/raw/<slug>_18_agent/`

### 5. (Future) Multi-cluster LLM-coarse
- Not yet evaluated. The `rules/financebench_multi_clusters/llm/gpt54/one_shot/` directory exists but eval_merge for it is empty.

---

## Headline experiments and takeaways

| Question | Winner on sAcc | Winner on uAcc | Winner on cost_u |
|----------|----------------|----------------|------------------|
| Single cluster, accuracy | **LLM-coarse gpt54** (0.910) | **LLM-coarse gpt54** (0.892) | **Agent-refined gpt54** (0.036) |
| Multi cluster, accuracy | LLM-coarse-multi not measured | **Agent-raw multi gpt54** (0.843) | **Agent-raw multi opus47** (0.013) |
| Best single-cluster generalization gap (sAcc − uAcc) | LLM-coarse: **+0.018** | Agent-raw single gpt54: +0.106 | Multi-cluster gpt54 agent: +0.013 (essentially tied) |

### Surprising findings

1. **LLM-coarse beats agent on accuracy** on single cluster — counterintuitive given the agent has access to feedback loops. The agent's iterative refinement saves cost but loses accuracy. Suggests gpt54's single-shot rule synthesis is already strong for this task; agentic refinement helps only on cost.

2. **Multi-cluster generalizes much better than single-cluster** — Agent-raw multi-gpt54: sAcc 0.856, uAcc 0.843, gap +0.013. The 18-doc diverse sample is enough to learn rules that transfer to other document types. Single-cluster (10 same-type docs) overfits more.

3. **Opus47 vs gpt54 for agent generation**:
   - Single cluster: opus47 slightly better on both accuracy and uAcc, slightly worse on cost.
   - Multi cluster: opus47 much cheaper (10×) but loses 0.14 uAcc to gpt54. Opus produces tighter rules; gpt54 produces more conservative/broader rules.

4. **Agent-refined regresses on unsampled** — refinement at generation time over-prunes. The downstream selection algorithms (Pareto v2, fallback) handle this trade-off better.

---

## Dataset details

| Cluster | # Sampled docs | # Unsampled docs | # Questions | Total |
|---------|--------------:|----------------:|------------:|------:|
| `financebench_single_cluster` | 10 | 50 | 10 | 600 doc-question pairs |
| `financebench_multi_clusters` | 18 | 96 | 12 | 1,368 doc-question pairs |

- **Single cluster** = all docs are 10-K filings.
- **Multi cluster** = mixed: 10-K, 10-Q, 8-K, earnings releases.

Sampled docs (with ground truth) live in `data/financebench/sample_doc_labels.json`. Unsampled docs (held-out) in `data/financebench/unsampled_doc_labels.json`.

---

## How to add a new generation strategy

1. Create a new module under `src/rule_gen_<name>.py` (or extend an agent prompt in `task_prompt_*.py`).
2. Write rules to `rules/financebench_{single,multi}_cluster{s}/<gen_type>/<model>/<variant>/<slug>_{N}_{type}/`.
3. Run `test/run_eval_merge_sampled.py` and `run_eval_merge_unsampled.py` against the new pool to populate `eval_merge/`.
4. Add a row to the summary table above and a method-description section.

---

## File index

| File | Purpose |
|------|---------|
| `src/rule_gen_llm_coarse.py` | LLM-coarse single-shot generator |
| `src/rule_gen_agent.py` | Earlier agent generator (interactive) |
| `src/rule_gen_agent_claude.py` | Current agent generator wrapped around `claude -p` |
| `LSF/task_prompt_rule_gen.py` | Task-prompt builder for agent rule generation |
| `test/run_eval_merge_sampled.py` | Evaluates the full pool on 10 sampled docs |
| `test/run_eval_merge_unsampled.py` | Evaluates the full pool on 50 unsampled docs |
| `docs/rule_gen_llm_coarse.md` | Spec for LLM-coarse |
| `docs/rule_gen_agent.md`, `rule_gen_agent_coarse.md`, `rule_gen_agent_exact.md` | Agent specs |
