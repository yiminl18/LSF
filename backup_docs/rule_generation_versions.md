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
| **Task 1 — Agentic-gen (random)** | `agent/run_agent_gen.py --sample-set random` | single | opus47 | 10 | 0.960 | 0.0062 | 10 | 0.820 | 0.0046 |
| **Task 2 — Agentic-gen (FPS)** | `agent/run_agent_gen.py --sample-set fps` | single (FPS) | opus47 | 10 | 0.960 | 0.0082 | 10 | 0.786 | 0.0079 |
| LLM-coarse | `src/rule_gen_llm_coarse.py` | multi | gpt54 | — | — | — | — | — | — (not yet evaluated) |

> Dataset: FinanceBench. "single cluster" = sample drawn from 60 FinanceBench docs (10 sampled, 50 unsampled). "multi cluster" = 18 sampled, 96 unsampled (12 questions instead of 10).

---

## Method descriptions

### 1. LLM-coarse (`src/rule_gen_llm_coarse.py`)
- **Approach**: single prompt to gpt54 with the question + a few example documents. The model emits a Python span-retrieval function in one shot.
- **Result on single cluster**: 100 rules generated per question on average. Full-pool merge: **sAcc=0.91, uAcc=0.89**. Highest accuracy on both splits — but also highest retrieval cost (~17% of doc tokens). The "default" rule pool everything else is compared against.
- **Output dir**: `rules/financebench/lsf/single_cluster/llm/gpt54/one_shot/<slug>_10_llm/`
- **Spec**: `docs/rule_gen_llm_coarse.md`

### 2. Agent-raw (`src/rule_gen_agent_claude.py` / `src/rule_gen_agent_langchain.py`)
- **Approach**: Claude Code (`claude -p`) opens an interactive session with full tool access (file read/write, Python exec). Iteratively builds rules, evaluates them, refines until target merge accuracy or budget exhaustion.
- **Result on single cluster (gpt54-generated)**: sAcc=0.86, uAcc=0.75. Lower accuracy than LLM-coarse but **~35% cheaper retrieval** (cost_s=0.115 vs 0.172). The agent compresses rules during generation.
- **Result on single cluster (opus47-generated)**: sAcc=0.89, uAcc=0.78. Better than gpt54-generated agent rules on both axes, slightly worse on cost (0.13 vs 0.12).
- **Output dirs**:
  - `rules/financebench/lsf/single_cluster/agent/gpt54/raw/<slug>_10_agent/`
  - `rules/financebench/lsf/single_cluster/agent/opus47/raw/<slug>_10_agent/`
- **Spec**: `docs/rule_gen_agent.md`

### 3. Agent-refined (post-process pass on Agent-raw)
- **Approach**: a second agent pass takes Agent-raw rules and merges/specializes them to reduce retrieval cost while preserving sampled accuracy.
- **Result on single cluster (gpt54)**: sAcc=0.88 (small drop from 0.86 base — actually slight improvement), uAcc=**0.734** (regression from raw 0.754). cost=**0.036** (3× cheaper than raw).
- **Trade-off**: large cost win on sampled (essentially matches sampled-refined cost ratios from algorithmic refinement); modest uAcc regression. Overfitting risk: refinement on the sampled-only signal compresses too aggressively.
- **Output dir**: `rules/financebench/lsf/single_cluster/agent/gpt54/refined/<slug>_10_agent_refined/`

### 4. Multi-cluster Agent-raw
- **Approach**: same agent rule-generation pipeline applied to the 18-doc multi-cluster sample set (covers 10-K, 10-Q, 8-K, earnings releases).
- **Result on multi-cluster (gpt54)**: sAcc=0.86, **uAcc=0.84** (much smaller sampled→unsampled gap than single-cluster's 0.86→0.75). Diverse sampled docs apparently produce more generalizable rules.
- **Result on multi-cluster (opus47)**: sAcc=0.85, uAcc=0.71, cost=**0.013** (10× cheaper than gpt54-multi). Opus rules are compact but less robust on unsampled.
- **Output dirs**: `rules/financebench/lsf/multi_clusters/agent/{gpt54,opus47}/raw/<slug>_18_agent/`

### 5. (Future) Multi-cluster LLM-coarse
- Not yet evaluated. The `rules/financebench/lsf/multi_clusters/llm/gpt54/one_shot/` directory exists but eval_merge for it is empty.

### 6. Agentic-gen (`agent/run_agent_gen.py`)
- **Approach**: Claude Opus 4.7 generates rules from scratch by inspecting the reconstructed JSON of each sampled doc (via `list_docs` + `read_doc_json`), authoring new Python rule functions (via `write_rule`), and grounding each rewrite step in the verification tools (`compute_cost`, `verify_accuracy`, `list_rules`, `inspect_rule`). Hard constraint: `match_rate = 1.0` on every sampled doc, checked via `verify_accuracy --d-star-mode all_labeled`. Budget: 30 verify_accuracy calls per question. **No rule refinement** is applied; the table values are for the raw generated pool. Typical output: 1–2 rules per question.
- **Spec**: `docs/rule_generation_agentic_from_pdf.md` (the agent reads reconstructed JSON, never PDFs).
- **Task 1 (random sample)**: rules generated from `data/financebench/sample/single_cluster/random/sample_doc_labels.json` (original random 10-doc sample).
  - **Result**: **sAcc=0.960, uAcc=0.820**, cost_s=**0.0062**, cost_u=**0.0046**.
  - Highest sAcc of any single-cluster method. Cost is ~37× cheaper than LLM-coarse on unsampled (0.0046 vs 0.1686).
  - Output dir: `rules/financebench/lsf/single_cluster/agent/opus47/agentic/raw/<slug>_10_agentic/`
- **Task 2 (FPS sample)**: rules generated from `data/financebench/sample/single_cluster/fps/sample_doc_labels.json` (10 docs selected by Farthest-Point Sampling on document embeddings, chosen to maximise structural diversity).
  - **Result**: **sAcc=0.960, uAcc=0.786**, cost_s=0.0082, cost_u=0.0079.
  - Same sAcc as Task 1. FPS did **not** improve unsampled generalisation over the random sample (uAcc 0.786 vs 0.820) — the random sample was already diverse enough for these 10 questions.
  - Output dir: `rules/financebench/lsf/single_cluster/agent/opus47/agentic_fps/raw/<slug>_10_agentic_fps/`
- **Status**: complete. Both tasks evaluated via `test/run_eval_merge_agentic.py`.

---

## Headline experiments and takeaways

| Question | Winner on sAcc | Winner on uAcc | Winner on cost_u |
|----------|----------------|----------------|------------------|
| Single cluster, accuracy | **Agentic-gen Task 1 & 2** (0.960) | **LLM-coarse gpt54** (0.892) | **Agentic-gen Task 1** (0.0046) |
| Single cluster, best accuracy+cost joint | **Agentic-gen Task 1** (sAcc 0.960, cost 0.0046) | — | — |
| Multi cluster, accuracy | LLM-coarse-multi not measured | **Agent-raw multi gpt54** (0.843) | **Agent-raw multi opus47** (0.013) |
| Best single-cluster generalization gap (sAcc − uAcc) | LLM-coarse: **+0.018** | Agentic-gen Task 1: **+0.140** | Agentic-gen Task 1 cost_u: **0.0046** |

### Surprising findings

1. **Agentic-gen (JSON loop) beats LLM-coarse on sampled accuracy** — sAcc 0.960 vs 0.910, while being ~37× cheaper on retrieval cost (cost_u 0.0046 vs 0.1686). The verify-accuracy feedback loop during generation lets Opus 4.7 write tighter rules that still cover all training docs, rather than the broad over-coverage produced by LLM-coarse's single-shot approach.

2. **FPS sampling did not improve unsampled generalization** — Task 2 (FPS) uAcc=0.786 vs Task 1 (random) uAcc=0.820. Farthest-Point Sampling was designed to maximise structural diversity in D_s; for these 10 cover-page / financial-statement questions the random sample was already diverse enough. FPS may help more on questions that require multi-document layout variation.

3. **LLM-coarse beats agent on unsampled accuracy** — LLM-coarse uAcc=0.892 vs Agentic-gen Task 1 uAcc=0.820. The generalization gap for Agentic-gen is larger (0.14 vs 0.018 for LLM-coarse). The hard constraint (`match_rate = 1.0` on all sampled docs) may cause the agent to overfit to the 10 training docs.

4. **Multi-cluster generalizes much better than single-cluster** — Agent-raw multi-gpt54: sAcc 0.856, uAcc 0.843, gap +0.013. The 18-doc diverse sample is enough to learn rules that transfer to other document types. Single-cluster (10 same-type docs) overfits more.

5. **Opus47 vs gpt54 for agent generation**:
   - Single cluster: opus47 slightly better on both accuracy and uAcc, slightly worse on cost.
   - Multi cluster: opus47 much cheaper (10×) but loses 0.14 uAcc to gpt54. Opus produces tighter rules; gpt54 produces more conservative/broader rules.

6. **Agent-refined regresses on unsampled** — refinement at generation time over-prunes. The downstream selection algorithms (Pareto v2, fallback) handle this trade-off better.

---

## Dataset details

| Cluster | Sample set | # Sampled docs | # Unsampled docs | # Questions | Total |
|---------|------------|---------------:|-----------------:|------------:|------:|
| `financebench/lsf/single_cluster` | random | 10 | 50 | 10 | 600 doc-question pairs |
| `financebench/lsf/single_cluster` | FPS | 10 | 50 | 10 | 600 doc-question pairs |
| `financebench/lsf/multi_clusters` | — | 18 | 96 | 12 | 1,368 doc-question pairs |

- **Single cluster** = all docs are 10-K filings.
- **Multi cluster** = mixed: 10-K, 10-Q, 8-K, earnings releases.
- **FPS sample** = 10 docs selected by Farthest-Point Sampling on document embeddings to maximise structural diversity; unsampled set is the remaining 50 docs not in the FPS sample.

| Split | Labels file |
|-------|-------------|
| Random sampled | `data/financebench/sample/single_cluster/random/sample_doc_labels.json` |
| Random unsampled | `data/financebench/sample/single_cluster/random/unsampled_doc_labels.json` |
| FPS sampled | `data/financebench/sample/single_cluster/fps/sample_doc_labels.json` |
| FPS unsampled | `data/financebench/sample/single_cluster/fps/unsampled_doc_labels.json` |

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
| `src/rule_gen_agent_langchain.py` | Earlier agent generator (interactive) |
| `src/rule_gen_agent_claude.py` | Current agent generator wrapped around `claude -p` |
| `LSF/task_prompt_rule_gen.py` | Task-prompt builder for agent rule generation |
| `test/run_eval_merge_sampled.py` | Evaluates the full pool on 10 sampled docs |
| `test/run_eval_merge_unsampled.py` | Evaluates the full pool on 50 unsampled docs |
| `docs/rule_gen_llm_coarse.md` | Spec for LLM-coarse |
| `docs/rule_gen_agent.md`, `rule_gen_agent_coarse.md`, `rule_gen_agent_exact.md` | Agent specs |
