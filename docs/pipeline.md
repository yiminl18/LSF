# End-to-End Rule Pipeline — `test/rule_end_to_end.py`

---

## Overview

This script runs the full rule-based QA pipeline for a set of questions on sampled and unsampled documents. It chains four stages: rule generation → (optional) rule refinement → rule application → evaluation. All outputs follow the same file formats as the individual module results already stored under `results/`.

---

## Pipeline Stages

```
[1] Rule Generation
    rule_gen_llm_coarse  |  rule_gen_agent_coarse  |  any src/rule_gen/*.py
         ↓
[2] (Optional) Rule Refinement
    src/rule_refine/v1.py
         ↓
[3] Rule Application (sampled + unsampled docs)
    src/rule_apply/merge.py
         ↓
[4] Evaluation (accuracy, cost ratio, latency)
    LLM-as-judge → results/
```

---

## CLI Interface

```bash
python test/rule_end_to_end.py \
    --rule-gen-module   src/rule_gen/llm_coarse.py \
    --queries-file      data/financebench/sample_queries.txt \
    --sample-labels     data/financebench/sample/single_cluster/random/sample_doc_labels.json \
    --unsampled-labels  data/financebench/sample/single_cluster/random/unsampled_doc_labels.json \
    --processing-dir    data/financebench/processing \
    --rules-dir         rules/llm/financebench \
    --output-dir        results/e2e \
    [--use-refine] \
    [--skip-existing]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--rule-gen-module` | `src/rule_gen/llm_coarse.py` | Path to any `src/rule_gen/*.py` file — the pipeline dynamically imports its `rule_gen_*` function |
| `--queries-file` | `data/financebench/sample_queries.txt` | Questions to run |
| `--sample-labels` | `data/financebench/sample/single_cluster/random/sample_doc_labels.json` | Sampled doc labels (`"DOCNAME.pdf" → {q: a}`) |
| `--unsampled-labels` | `data/financebench/sample/single_cluster/random/unsampled_doc_labels.json` | Unsampled doc labels |
| `--processing-dir` | `data/financebench/processing` | Directory of `*_reconstructed.json` files |
| `--rules-dir` | `rules/llm/financebench` | Where generated rule `.py` files are stored |
| `--output-dir` | `results/e2e` | Root output directory for this pipeline run |
| `--use-refine` | off | If set, run `rule_refine` between generation and application |
| `--skip-existing` | off | If set, skip any stage whose output already exists on disk |

---

## Stage 1 — Rule Generation

**Module:** any file passed to `--rule-gen-module` whose filename matches `rule_gen_*.py` and exports a function `rule_gen_*(documents, question, ground_truth, ...) -> dict`.

The pipeline:
1. Reads all questions from `--queries-file`
2. Loads the 10 sampled docs from `--sample-labels` (used as training signal for rule gen)
3. For each question, checks if a rule folder already exists at `{rules_dir}/{question_slug}_10_llm/` — skips if `--skip-existing` and folder present
4. Calls the generation function and stores rules in `{rules_dir}/{question_slug}_10_llm/`

**Output:** rule `.py` files in `{rules_dir}/{question_slug}_10_llm/`

Rule gen result JSON written to: `{output_dir}/rule_gen/{question_slug}_rule_gen.json`

---

## Stage 2 — Rule Refinement (optional, `--use-refine`)

If `--use-refine` is set:
1. Loads target accuracy from Stage 3's sampled eval (or computes it inline)
2. Calls `rule_refine(rule_names, target_accuracy, question, ...)` for each question
3. Stores selected rule `.py` files to `{output_dir}/refined_rules/{question_slug}/`
4. Writes refinement result to `{output_dir}/rule_refine/{question_slug}_refine.json`

The effective `rules_dir` for Stage 3 becomes `{output_dir}/refined_rules/` if `--use-refine` is set, otherwise `{rules_dir}`.

---

## Stage 3 — Rule Application

Calls `rule_apply_merge` for each (question, doc) pair on both splits.

**Sampled docs (10):** from `--sample-labels`
**Unsampled docs (50):** from `--unsampled-labels`

Intermediate prediction files written to:
```
{output_dir}/rule_run/merge/{question_slug}/{rule_set_slug}_merge.json        # sampled
{output_dir}/rule_run/merge_unsampled/{question_slug}/{rule_set_slug}_merge.json  # unsampled
```

---

## Stage 4 — Evaluation

For each (question, split), evaluate predictions with LLM-as-judge and compute metrics.

**Per-question per-split eval file:** `{output_dir}/eval/{question_slug}_{split}.json`

Schema (matches `results/eval_merge_all/` format exactly):
```json
{
  "question": "...",
  "question_slug": "..._10",
  "split": "sampled",
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
      "input_tokens": 123
    }
  ]
}
```

**Summary file:** `{output_dir}/eval/summary.json`

Schema (matches `results/eval_merge_all/summary.json`):
```json
[
  {
    "question": "...",
    "question_slug": "..._10",
    "sampled":   { "n": 10, "accuracy": 0.9, "avg_cost_ratio": 0.0021, ... },
    "unsampled": { "n": 50, "accuracy": 0.88, "avg_cost_ratio": 0.0019, ... }
  }
]
```

---

## Full Output Directory Structure

```
{output_dir}/                                  e.g. results/e2e/
├── rule_gen/
│   ├── {question_slug}_rule_gen.json          stage 1 result per question
│   └── ...
├── refined_rules/                             only if --use-refine
│   ├── {question_slug}/
│   │   ├── rule_exact_name_h1.py
│   │   └── ...
│   └── ...
├── rule_refine/                               only if --use-refine
│   ├── {question_slug}_refine.json
│   └── ...
├── rule_run/
│   └── merge/
│       └── {question_slug}/
│           └── {rule_set_slug}_merge.json
├── rule_run_unsampled/
│   └── merge/
│       └── {question_slug}/
│           └── {rule_set_slug}_merge.json
├── eval/
│   ├── {question_slug}_sampled.json
│   ├── {question_slug}_unsampled.json
│   └── summary.json
└── pipeline_summary.json                      full run metadata
```

### Pipeline summary file

`{output_dir}/pipeline_summary.json`:
```json
{
  "timestamp": "2026-04-30T10:00:00Z",
  "rule_gen_module": "src/rule_gen/llm_coarse.py",
  "use_refine": false,
  "queries_file": "data/financebench/sample_queries.txt",
  "num_questions": 10,
  "num_sampled_docs": 10,
  "num_unsampled_docs": 50,
  "questions": [
    {
      "question": "What is the registrant's exact name?",
      "question_slug": "what_is_the_registrants_exact_name_10",
      "num_rules_generated": 25,
      "num_rules_after_refine": null,
      "sampled_accuracy": 0.9,
      "unsampled_accuracy": 0.98,
      "avg_cost_ratio_sampled": 0.0021,
      "avg_cost_ratio_unsampled": 0.0019
    }
  ],
  "overall": {
    "avg_sampled_accuracy": 0.84,
    "avg_unsampled_accuracy": 0.81,
    "avg_cost_ratio": 0.031
  }
}
```

---

## Edge Cases

| Situation | Behavior |
|---|---|
| Rule gen already ran (`--skip-existing`) | Skip gen; load existing rules from `rules_dir` |
| No rule folder for a question | Skip all downstream stages for that question; log warning |
| `--use-refine` but refine output exists (`--skip-existing`) | Skip refine; load existing refined rules |
| Doc JSON missing from processing dir | Skip that doc; record in per_doc as `predicted=null, correct=false` |
| Stage fails for one question | Log error and continue to next question |

---

## Relation to Other Modules

| Module | Stage | Role |
|---|---|---|
| `src/rule_gen/llm_coarse.py` | 1 | Default rule generator |
| `src/rule_gen/agent_coarse.py` | 1 | Alternative rule generator |
| `src/rule_refine/v1.py` | 2 | Optional rule subset selection |
| `src/rule_apply/merge.py` | 3 | Apply rules, union spans, call LLM |
| `src/eval_rule.py` | 4 | LLM-as-judge evaluation logic |
