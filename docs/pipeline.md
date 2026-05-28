# End-to-End Rule Pipeline — `test/rule_end_to_end.py`

---

## Overview

This script runs the full rule-based QA pipeline for a set of questions on sampled and unsampled documents. It chains **four** stages: document sampling → rule generation → (optional) rule refinement → rule application + evaluation. Every stage exposes a strategy choice, so a single CLI invocation can mix-and-match the documented approaches in the four `docs/approach/*.md` files.

### Data source — reconstructed JSON

Every stage in this pipeline operates on **reconstructed JSON** documents, not raw text/PDF:

```
data/<dataset>/processing/<DOC_NAME>_reconstructed.json
```

Each file is a single dict with a `texts` list of span dicts (fields: `text`, `label`, `page_no`, `bold`, `size`, `structure.level`, `structure.path_text`, optional `table_data.cells`, …). Span-retrieval rules in `rule_<name>(doc) -> list[dict]` operate directly on `doc["texts"]`. There is no PDF or raw-text ingestion inside the pipeline — that's done upstream by the doc-reconstruction step, which is out of scope for this doc.

If a strategy expects raw `.txt` (e.g., the legacy `agentic_rule_full_data` baseline in `docs/approach/rule_end_to_end.md`), it is **not** wired into this pipeline.

### Dataset stats (docs + queries in scope)

Each dataset has a canonical source for the pipeline. Counts come from the file paths shown:

| Dataset | Docs | Queries | Data source | Query source |
|---|---:|---:|---|---|
| **financebench** | **100** (20 sampled + 80 unsampled) | **12** | `data/financebench/sample/multi_cluster/random/{sample,unsampled}_doc_labels.json` | `data/financebench/multi_cluster_queries.txt` |
| **court** | **294** | **13** | `data/court/all_labels.json` | `data/court/queries.json` |
| **nopv** | **242** | **12** | `data/nopv/all_labels.json` | `data/nopv/queries.json` |
| **officeqa** | **200** | **16** | `data/officeqa/all_labels.json` | `data/officeqa/queries.json` |

For FinanceBench the **multi_cluster** split is the canonical one — single_cluster's 10/50 split is a legacy single-doc-type benchmark and is not used by this pipeline going forward. For Court / NOPV / OfficeQA the labels file is flat and Stage 1 (sampling) creates the sampled / unsampled split at run time using the chosen strategy and the size cap.

### Per-dataset data source and query source

| Dataset | Document JSON dir | Labels file (all docs) | Queries file | Total docs | Q |
|---|---|---|---|---:|---:|
| **financebench** | `data/financebench/processing/<DOC>_reconstructed.json` | `data/financebench/sample/<cluster>/<sampler>/sample_doc_labels.json` + `unsampled_doc_labels.json` (split-aware) | `data/financebench/sample_queries.txt` (single_cluster, 10 Q) or `data/financebench/multi_cluster_queries.txt` (multi_cluster, 12 Q) | 141 in `processing/`; 60 / 100 in checked-in benchmark splits | 10 / 12 |
| **court** | `data/court/processing/<DOC>_reconstructed.json` (must be built before this pipeline can run on court) | `data/court/all_labels.json` (single file, no pre-split) | `data/court/queries.json` (list of `{text, answer_type}`) | 294 (labels) / 300 (raw `.txt`) | 13 |
| **nopv** | `data/nopv/processing/<DOC>_reconstructed.json` (must be built before this pipeline can run on nopv) | `data/nopv/all_labels.json` (single file, no pre-split) | `data/nopv/queries.json` (list of `{text, answer_type}`) | 242 (labels) / 250 (raw `.txt`) | 12 |
| **officeqa** | `data/officeqa/processing/<DOC>_reconstructed.json` (must be built before this pipeline can run on officeqa) | `data/officeqa/all_labels.json` (single file, no pre-split) | `data/officeqa/queries.json` (list of `{text, answer_type}`) | 200 (labels) / 697 (raw `.txt`) | 16 |

**Labels file format**

- FinanceBench: pre-split into `sample_doc_labels.json` + `unsampled_doc_labels.json` per cluster + sampler. Each is `{"<DOC>.pdf": {<question>: <answer>}}`.
- Court / NOPV / OfficeQA: single flat file `data/<dataset>/all_labels.json` of the same shape. The pipeline's Stage 1 (sampling) splits it into sampled / unsampled at run time using the chosen strategy and the size cap.

**Queries file format**

- FinanceBench: plain `.txt`, one question per line.
- Court / NOPV / OfficeQA: JSON list of objects with `{"text": <question>, "answer_type": <"string" | "list[string]" | …>}`. The pipeline reads the `text` field as the question and may pass `answer_type` to downstream judges that support it.

**Reconciling label and `.txt` counts:** for court/nopv/officeqa the raw `text/` directory has more docs than `all_labels.json` (e.g., 300 vs 294 for court). Only docs present in `all_labels.json` are in scope for this pipeline; unlabeled `.txt` files are ignored.

```
[1] Sampling           docs/approach/sampling.md
       ↓ picks K sampled docs + leaves the rest as "unsampled"
[2] Rule Generation    docs/approach/rule_generation.md
       ↓ produces a rule pool per question from the sampled docs
[3] Rule Refinement    docs/approach/rule_refinement.md     (optional)
       ↓ selects a smaller subset of the pool
[4] Rule Application   docs/approach/rule_application.md
       ↓ applies the (refined) rule set to every doc + LLM judge
   Evaluation          accuracy, cost ratio, latency on sampled + unsampled
```

Each stage's strategy can be selected independently. The pipeline does the wiring (paths, naming, output layout) so individual approaches stay decoupled.

---

## Pipeline Stages and Strategies

### Stage 1 — Sampling

Pick which docs go into the **sampled** split (used as training signal for rule gen and as the in-loop oracle for refinement). The rest of the corpus becomes **unsampled** for evaluation.

| Strategy | Code | Notes |
|---|---|---|
| `random` ⭐ | (pre-built JSONs in `data/<dataset>/sample/<cluster>/random/`) | Sample/unsampled split is fixed once per dataset/cluster — just point at the existing label files. Fine for uniform corpora. Included in the test grid. |
| `fps` ⭐ | `src/sampling/run_fps_sampling.py` | Farthest-Point Sampling on contrast-normalised similarity-curve vectors. Hyperparameter-free; elbow self-detects K. Deterministic cluster-coverage guarantee when features are well-separated — picks rare-cluster docs *first*, not last. Included in the test grid. |

See `docs/approach/sampling.md` for the algorithm spec.

**Recommended:** `fps`. Random sampling can miss rare clusters (probability `(1−s/n)^k` for k under-represented clusters); FPS picks at least one rep per cluster by construction when separation holds.

#### Sample-size cap

The sampled split must contain **at most 20 docs** for every dataset, regardless of corpus size. The cap is a hard ceiling on rule-gen and refinement cost. Twenty docs is generally enough variety to capture the cluster structure of the corpora used in this pipeline without blowing the verify-accuracy budget downstream.

`fps` may **stop early** with fewer than 20 picks: if its elbow rule (`g_{i+1} < 0.5 · g_i`) fires at `K < 20`, that's the algorithmic signal that further picks don't add new cluster structure — keep the smaller sample and move on. The 20-doc cap is the worst-case ceiling when the elbow never fires; it is not the target.

`random` should be configured to draw exactly 20 docs (or all docs if `N < 20`).

##### Recommended sampled count per dataset

| Dataset | `N` | Recommended sample size | Resulting unsampled count |
|---|---:|---:|---:|
| **financebench** (multi_cluster) | 100 | **20** (cap) | 80 |
| **court** (`all_labels.json`) | 294 | **20** (cap) | 274 |
| **nopv** (`all_labels.json`) | 242 | **20** (cap) | 222 |
| **officeqa** (`all_labels.json`) | 200 | **20** (cap) | 180 |

FinanceBench's checked-in `data/financebench/sample/multi_cluster/random/` already uses a 20-doc sample. For Court / NOPV / OfficeQA the pipeline reads `data/<dataset>/all_labels.json` and Stage 1 produces the split at run time. Court / NOPV / OfficeQA currently have raw `.txt` only — the JSON-based stages of this pipeline cannot run on them until `_reconstructed.json` files are built; the `agentic_rule_full_data` baseline (see `docs/approach/rule_end_to_end.md`) operates directly on the `.txt` files and is not part of this pipeline.

### Stage 2 — Rule Generation

For each question, produce a rule pool from the sampled docs.

| Strategy | Code | Notes |
|---|---|---|
| `llm_coarse` | `src/rule_gen/llm_coarse.py` | Single-shot LLM generates ~100 broad rules. Best uAcc on its own (0.892). Backwards-compatible alias for `llm_coarse_gpt54`. |
| `llm_coarse_gpt54` ⭐ | `src/rule_gen/llm_coarse.py` (model=gpt54) | LLM-coarse with full gpt-5.4. Included in the test grid. |
| `llm_coarse_gpt54mini` ⭐ | `src/rule_gen/llm_coarse.py` (model=gpt54mini) | LLM-coarse with gpt-5.4-mini. Included in the test grid. |
| `agent_langchain` | `src/rule_gen/agent_langchain.py` | LangChain AgentExecutor with substring-match feedback + LLM union check. |
| `agent_claude` | `src/rule_gen/agent_claude.py` (driver: `agent/run_agent_gen.py`) | Claude Opus 4.7 free-form agent. ~5–10 rules/Q, highest sAcc. |
| `agent_codex` | `src/rule_gen/agent_codex.py` | Codex equivalent of `agent_claude` with the same prompt. Backwards-compatible alias for `agent_codex_gpt54`. |
| `agent_codex_gpt54` ⭐ | `src/rule_gen/agent_codex.py` (model=gpt54) | Agentic Codex generation with gpt-5.4. Included in the test grid. |
| `agent_codex_gpt54mini` ⭐ | `src/rule_gen/agent_codex.py` (model=gpt54mini) | Agentic Codex generation with gpt-5.4-mini. Included in the test grid. |

See `docs/approach/rule_generation.md` for full per-strategy results.

**Recommended (in the test grid):**
- `llm_coarse` for **breadth + best generalization on its own** (single LLM call, ~100 rules, 0.892 uAcc without any refinement).
- `agent_codex` for **focused agentic rule sets via Codex/gpt-5.4** (~5–10 rules/Q, same prompt as `agent_claude`, no Claude Code dependency on the gen side).

### Stage 3 — Rule Refinement (optional)

Take the Stage 2 pool and select a smaller subset.

| Strategy | Code | Notes |
|---|---|---|
| `v1` | `src/rule_refine/v1.py` | Cost-sort + exponential search + backward prune. |
| `p_mini` ⭐ | `src/rule_refine/selection/select_rules_pareto.py` (MODEL_NAME=gpt54mini) | Pareto greedy + cheap judge. Lowest cost, smallest overfit gap. |
| `p_hybrid` ⭐ | `src/rule_refine/selection/select_rules_pareto_hybrid.py` | Hybrid: **gpt-5.4-mini** for coverage estimation (sort key), **gpt-5.4** for in-loop admission + final merge verification. Cheap signal where it's good, strong signal where correctness matters. Included in the test grid. |
| `p_gpt54` | `src/rule_refine/selection/select_rules_pareto.py` (MODEL_NAME=gpt54) | Pareto greedy + strong judge. |
| `p_proxy` | `src/rule_refine/selection/select_rules_pareto_proxy.py` | Pareto greedy + substring-only judge (no LLM). |
| `p_v2` | `src/rule_refine/selection/select_rules_pareto_v2.py` | p_gpt54 + accuracy floor + backward prune. |
| `p_v3` | `src/rule_refine/selection/select_rules_pareto_v3.py` | p_v2 + cumulative-prefix fallback for hard questions. |
| `agentic` | `src/rule_refine/agentic.py` | Claude Opus 4.7 selector. Highest measured uAcc (0.870), fewest rules. Replaced by `agentic_codex` in the test grid to keep the pipeline Claude-free. |
| `agentic_codex` | `src/rule_refine/agentic_codex.py` | Codex equivalent of `agentic` — same prompt and constraints. Backwards-compatible alias for `agentic_codex_gpt54`. |
| `agentic_codex_gpt54` ⭐ | `src/rule_refine/agentic_codex.py --model gpt54` | Codex selector with gpt-5.4 as the agent backbone. Included in the test grid. |
| `agentic_codex_gpt54mini` ⭐ | `src/rule_refine/agentic_codex.py --model gpt54mini` | Codex selector with gpt-5.4-mini as the agent backbone — cheaper, more variable picks. Included in the test grid. |

Skip Stage 3 entirely with `--refine-strategy none` to feed the full Stage 2 pool directly into Stage 4. See `docs/approach/rule_refinement.md` for results.

**Recommended:**
- `p_mini` for **cheap, regularizing selection** (smallest overfit gap, cheapest cost — runs in minutes).
- `agentic_codex` for **agentic selection without a Claude dependency** — uses gpt-5.4 via Codex with the same prompt as `agentic`. (`agentic` itself remains available if you have Claude Code; it has the only measured uAcc numbers in this stage.)

### Stage 4 — Rule Application + Evaluation

Apply the rule set (refined if Stage 3 ran, else the full Stage 2 pool) to every doc and judge the answer.

| Strategy | Code | Notes |
|---|---|---|
| `merge` ⭐ | `src/rule_apply/merge.py` | Apply all rules in the set, union + dedupe spans, one LLM call per doc. Included in the test grid. |
| `default` ⭐ | `src/rule_apply/default.py` | Same as merge over the refined subset, but on a "NO" verdict from a cheap gpt54mini gate, re-merge over the full Stage 2 pool. Requires both a refined subset and the full pool. Included in the test grid. |

`individual` exists in the codebase for per-rule debugging but is not a pipeline strategy. See `docs/approach/rule_application.md`.

Evaluation is identical regardless of which apply strategy ran: `gpt54`-as-judge over the predicted vs. ground-truth answer, computing per-doc correctness, retrieved tokens, and cost ratio.

**Recommended (in the test grid):** both `merge` and `default` (recovers full-pool uAcc at fraction of cost via the cheap gpt54mini gate when Stage 3 ran).

---

## Test grid — 64 combinations

The recommended-strategy set across the four stages spans a **2 × 4 × 4 × 2 = 64** combinations test grid. Each stage's "recommended" axis now includes model variants where the underlying strategy supports both gpt-5.4 and gpt-5.4-mini:

| Stage | Recommended choices |
|---|---|
| Sampling | `random`, `fps` |
| Rule Generation | `llm_coarse_gpt54`, `llm_coarse_gpt54mini`, `agent_codex_gpt54`, `agent_codex_gpt54mini` |
| Rule Refinement | `p_mini`, `p_hybrid`, `agentic_codex_gpt54`, `agentic_codex_gpt54mini` |
| Rule Application | `merge`, `default` |

Notes on the new model-explicit variants:
- `llm_coarse_gpt54` / `llm_coarse_gpt54mini` — same `src/rule_gen/llm_coarse.py`, just `model_name=gpt54` vs `gpt54mini`.
- `agent_codex_gpt54` / `agent_codex_gpt54mini` — same `src/rule_gen/agent_codex.py`, just `--model gpt54` vs `gpt54mini`.
- `agentic_codex_gpt54` / `agentic_codex_gpt54mini` — same `src/rule_refine/agentic_codex.py`, just `--model gpt54` vs `gpt54mini`.
- `p_mini` is already gpt-5.4-mini by design; `p_hybrid` uses gpt-5.4-mini for coverage + gpt-5.4 for verify by design. Neither has a model toggle in the test grid.

### Sweeping the grid

```bash
for s in random fps; do
  for g in llm_coarse_gpt54 llm_coarse_gpt54mini agent_codex_gpt54 agent_codex_gpt54mini; do
    for r in p_mini p_hybrid agentic_codex_gpt54 agentic_codex_gpt54mini; do
      for a in merge default; do
        python src/pipeline.py \
          --sampling-strategy "$s" --rule-gen-strategy "$g" \
          --refine-strategy   "$r" --apply-strategy    "$a" \
          --dataset financebench --cluster multi_cluster \
          --output-dir "results/grid/${s}_${g}_${r}_${a}"
      done
    done
  done
done
```

---

## CLI Interface

```bash
python test/rule_end_to_end.py \
    --sampling-strategy   random \
    --rule-gen-strategy   llm_coarse \
    --refine-strategy     none \
    --apply-strategy      merge \
    --queries-file        data/financebench/sample_queries.txt \
    --dataset             financebench \
    --cluster             single_cluster \
    --processing-dir      data/financebench/processing \
    --output-dir          results/e2e \
    [--skip-existing]
```

### Arguments

| Argument | Default | Values | Description |
|---|---|---|---|
| `--sampling-strategy` | `random` | `random`, `fps` | Stage 1 strategy. `random` reuses pre-built label files under `data/<dataset>/sample/<cluster>/random/`. `fps` runs `src/sampling/run_fps_sampling.py` and writes a fresh `fps/` sub-directory. |
| `--rule-gen-strategy` | `llm_coarse` | `llm_coarse`, `agent_langchain`, `agent_claude`, `agent_codex` | Stage 2 strategy. The pipeline imports the matching module under `src/rule_gen/`. |
| `--refine-strategy` | `none` | `none`, `v1`, `p_mini`, `p_gpt54`, `p_proxy`, `p_v2`, `p_v3`, `agentic`, `agentic_codex` | Stage 3 strategy. `none` skips refinement and passes the Stage 2 pool through. |
| `--apply-strategy` | `merge` | `merge`, `default` | Stage 4 strategy. `default` requires a non-`none` refine strategy (it needs both a refined subset and a full pool to fall back to). |
| `--queries-file` | `data/financebench/sample_queries.txt` | path | Questions to run. |
| `--dataset` | `financebench` | `financebench`, `court`, `nopv`, `officeqa` | Selects the data/labels/rules root paths. |
| `--cluster` | `single_cluster` | `single_cluster`, `multi_cluster`, `all_docs` | Sub-directory under the dataset for label files. |
| `--processing-dir` | `data/<dataset>/processing` | path | Reconstructed-JSON dir. |
| `--output-dir` | `results/e2e` | path | Root output directory for this run. |
| `--skip-existing` | off | flag | Skip any stage whose output already exists on disk. |

Backwards-compatible legacy args still accepted:
- `--rule-gen-module <path>` — overrides `--rule-gen-strategy` by directly importing a `src/rule_gen/<name>.py` file.
- `--use-refine` — shorthand for `--refine-strategy v1` (kept for older shell scripts).

---

## Stage 1 — Sampling

**Default (`random`):** the pipeline just reads existing files at:
```
data/<dataset>/sample/<cluster>/random/sample_doc_labels.json
data/<dataset>/sample/<cluster>/random/unsampled_doc_labels.json
```

**FPS (`fps`):** the pipeline invokes `src/sampling/run_fps_sampling.py` which:
1. Merges sampled + unsampled label files into a single pool.
2. Embeds each doc in `L=50` span-binned chunks plus every query.
3. Builds contrast-normalised similarity-curve vectors `v_d`.
4. Runs farthest-point sampling with elbow stopping (`stop_ratio=0.5`, optional `max_K`).
5. Writes new `sample_doc_labels.json` / `unsampled_doc_labels.json` under `data/<dataset>/sample/<cluster>/fps/` and a diagnostic `fps_run.json`.

The pipeline picks up these new files for the rest of the run. The `embeddings.npz` cache means repeat FPS runs cost only the elbow search.

---

## Stage 2 — Rule Generation

For each question:
1. Load all sampled docs (from Stage 1's output).
2. Read this question's ground-truth answers per doc from the labels JSON.
3. Call the chosen `src/rule_gen/<strategy>.py` function and write its rule `.py` files to:
   ```
   rules/<dataset>/lsf/<cluster>/<strategy>/<model>/<variant>/<question_slug>_<N>_<tag>/
   ```
4. Save the rule-gen metadata JSON to `{output_dir}/rule_gen/{question_slug}_rule_gen.json`.

The exact `<tag>` and naming convention is defined by each generation approach (see `docs/approach/rule_generation.md`).

---

## Stage 3 — Rule Refinement (skip with `--refine-strategy none`)

If a refinement strategy is selected:
1. Load Stage 2's rule pool for each question.
2. Run the refine algorithm — its in-loop oracle (LLM judge or substring proxy) evaluates candidate subsets on the sampled docs.
3. Write the refined `rule_*.py` files to `{output_dir}/refined_rules/{question_slug}/`.
4. Save the refinement trace/metadata to `{output_dir}/rule_refine/{question_slug}_refine.json`.

The effective `rules_dir` Stage 4 sees becomes `{output_dir}/refined_rules/`. For `--apply-strategy default`, Stage 4 keeps a handle to the original Stage 2 pool for fallback.

---

## Stage 4 — Rule Application + Evaluation

Per (question, doc) pair on **both** splits (sampled and unsampled):
1. Call the chosen `src/rule_apply/<strategy>.py` function on the (possibly refined) rule set.
2. Run the LLM-as-judge on the predicted answer.
3. Write per-doc records to:
   ```
   {output_dir}/rule_run/{strategy}/{question_slug}/{rule_set_slug}.json             # sampled
   {output_dir}/rule_run_unsampled/{strategy}/{question_slug}/{rule_set_slug}.json   # unsampled
   ```
4. Aggregate to `{output_dir}/eval/{question_slug}_{split}.json`.

Per-doc and per-question schemas are identical regardless of `apply_strategy`; `apply_strategy = default` adds `used_fallback`, `relevance_verdict`, and separate token counts for the gate vs. answer model.

### Per-question per-split eval file

`{output_dir}/eval/{question_slug}_{split}.json`:

```json
{
  "question": "...",
  "question_slug": "..._10",
  "split": "sampled",
  "apply_strategy": "merge",
  "n": 10,
  "n_correct": 9,
  "accuracy": 0.9,
  "avg_latency": 1.16,
  "avg_retrieved": 35.9,
  "avg_input_tok": 120.0,
  "avg_cost_ratio": 0.0021,
  "per_doc": [
    {
      "doc_name": "AMCOR_2019_10K",
      "predicted": "Amcor plc",
      "ground_truth": "Amcor plc",
      "correct": true,
      "latency_seconds": 1.2,
      "retrieved_tokens": 42,
      "input_tokens": 123,
      "used_fallback": false
    }
  ]
}
```

### Summary file

`{output_dir}/eval/summary.json`:

```json
[
  {
    "question": "...",
    "question_slug": "..._10",
    "sampled":   { "n": 10, "accuracy": 0.9, "avg_cost_ratio": 0.0021, "...": "..." },
    "unsampled": { "n": 50, "accuracy": 0.88, "avg_cost_ratio": 0.0019, "...": "..." }
  }
]
```

---

## Full Output Directory Structure

```
{output_dir}/                                  e.g. results/e2e/
├── sampling/                                  only if --sampling-strategy fps
│   ├── sample_doc_labels.json
│   ├── unsampled_doc_labels.json
│   ├── fps_run.json
│   └── embeddings.npz
├── rule_gen/
│   └── {question_slug}_rule_gen.json
├── refined_rules/                             only if Stage 3 ran
│   └── {question_slug}/rule_*.py
├── rule_refine/                               only if Stage 3 ran
│   └── {question_slug}_refine.json
├── rule_run/
│   └── {apply_strategy}/{question_slug}/{rule_set_slug}.json
├── rule_run_unsampled/
│   └── {apply_strategy}/{question_slug}/{rule_set_slug}.json
├── eval/
│   ├── {question_slug}_sampled.json
│   ├── {question_slug}_unsampled.json
│   └── summary.json
└── pipeline_summary.json                      full run metadata
```

### `pipeline_summary.json`

```json
{
  "timestamp": "2026-05-27T16:30:00Z",
  "sampling_strategy":  "random",
  "rule_gen_strategy":  "llm_coarse",
  "refine_strategy":    "p_mini",
  "apply_strategy":     "default",
  "queries_file":       "data/financebench/sample_queries.txt",
  "dataset":            "financebench",
  "cluster":            "single_cluster",
  "num_questions":      10,
  "num_sampled_docs":   10,
  "num_unsampled_docs": 50,
  "questions": [
    {
      "question": "What is the registrant's exact name?",
      "question_slug": "what_is_the_registrants_exact_name_10",
      "num_rules_generated":     25,
      "num_rules_after_refine":  4,
      "sampled_accuracy":        0.9,
      "unsampled_accuracy":      0.98,
      "avg_cost_ratio_sampled":  0.0021,
      "avg_cost_ratio_unsampled":0.0019,
      "mean_fallback_rate":      0.10
    }
  ],
  "overall": {
    "avg_sampled_accuracy":   0.84,
    "avg_unsampled_accuracy": 0.81,
    "avg_cost_ratio":         0.031
  }
}
```

---

## Recipes

A few common combos:

```bash
# Baseline: random sampling + LLM-coarse + no refinement + merge
python test/rule_end_to_end.py \
  --sampling-strategy random --rule-gen-strategy llm_coarse \
  --refine-strategy none --apply-strategy merge

# Recommended cost-conscious deployment:
# random sampling, LLM-coarse pool, p_mini refinement, fallback application
python test/rule_end_to_end.py \
  --sampling-strategy random --rule-gen-strategy llm_coarse \
  --refine-strategy p_mini --apply-strategy default

# Recommended accuracy-conscious deployment:
# FPS sampling, Claude agent gen, agentic refinement, fallback application
python test/rule_end_to_end.py \
  --sampling-strategy fps --rule-gen-strategy agent_claude \
  --refine-strategy agentic --apply-strategy default
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| `--apply-strategy default` with `--refine-strategy none` | Error: default needs both refined and full pools. Either pick a refine strategy or switch to `--apply-strategy merge`. |
| Stage output already exists (`--skip-existing`) | Skip that stage; load existing artifacts from disk. |
| No rule folder for a question | Skip all downstream stages for that question; log warning. |
| `--sampling-strategy fps` but `fps_run.json` exists | Reuse cached embeddings + re-run elbow (cheap). |
| Doc JSON missing from processing dir | Skip that doc; per_doc record gets `predicted=null, correct=false`. |
| Stage fails for one question | Log error and continue to next question. |

---

## Relation to other docs

| Stage | Doc | Code package |
|---|---|---|
| 1 Sampling | `docs/approach/sampling.md` | `src/sampling/` |
| 2 Rule Generation | `docs/approach/rule_generation.md` | `src/rule_gen/` |
| 3 Rule Refinement | `docs/approach/rule_refinement.md` | `src/rule_refine/` |
| 4 Rule Application | `docs/approach/rule_application.md` | `src/rule_apply/` |
| End-to-end runner | this doc | `test/rule_end_to_end.py` |
