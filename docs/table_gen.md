# Table Generation Guide

This document describes how to generate performance and rule-generation status tables from LSF pipeline results. Tables are output as JSON.

---

## Dataset and Rule Type Mapping

### Dataset

| You say | Sampled labels | Unsampled labels |
|---|---|---|
| **single cluster doc** | `data/financebench/sample_doc_labels.json` | `data/financebench/unsampled_doc_labels.json` |
| **mix doc case** | `data/financebench/sample_mix_doc_labels.json` | `data/financebench/unsampled_mix_doc_labels.json` |

### Rule type

| You say | Rules directory | Results directory |
|---|---|---|
| **llm-gen rules** | `rules/llm/financebench/` | single cluster doc → `results/eval_merge_all/` <br> mix doc case → `results/e2e_mix_doc/eval/` |
| **agent gen rules** | `rules/agent/financebench_agent/` | single cluster doc → `results/e2e_agent/eval/` <br> mix doc case → `results/e2e_mix_doc_agent/eval/` |

---

## Generate Performance Table

**What it contains:** accuracy, average cost ratio, and average latency for each question on both sampled and unsampled documents.

**Source file:** `{results_dir}/summary.json`

Each entry in `summary.json` has this structure:
```json
{
  "question": "...",
  "question_slug": "...",
  "sampled": {
    "n": 10,
    "accuracy": 0.9,
    "avg_cost_ratio": 0.029,
    "avg_latency": 1.45
  },
  "unsampled": {
    "n": 50,
    "accuracy": 0.82,
    "avg_cost_ratio": 0.033,
    "avg_latency": 1.21
  }
}
```

**Output JSON schema:**
```json
[
  {
    "question": "What is total assets at year-end (from the audited balance sheet)?",
    "question_slug": "what_is_total_assets_at_yearend_from_the_audited_balance_she_10",
    "sampled_n": 10,
    "sampled_accuracy": 0.9,
    "sampled_avg_cost_ratio": 0.029318,
    "sampled_avg_latency": 1.45,
    "unsampled_n": 50,
    "unsampled_accuracy": 0.82,
    "unsampled_avg_cost_ratio": 0.033124,
    "unsampled_avg_latency": 1.21
  }
]
```

### Result directory lookup

| Dataset | Rule type | `summary.json` path |
|---|---|---|
| single cluster doc | llm-gen rules | `results/eval_merge_all/summary.json` |
| single cluster doc | agent gen rules | `results/e2e_agent/eval/summary.json` |
| mix doc case | llm-gen rules | `results/e2e_mix_doc/eval/summary.json` |
| mix doc case | agent gen rules | `results/e2e_mix_doc_agent/eval/summary.json` |

---

## Generate Rule Gen Status

**What it contains:** cost ratio and latency incurred during rule generation itself (not rule application), per question, on the sampled documents only.

### LLM-gen rules

Source files: `results/rule_gen_llm/{question_slug}_*.json` (one JSON per question, use the most recent by timestamp in the filename).

Relevant fields per file:
```json
{
  "question": "...",
  "question_slug": "...",
  "latency_seconds": 187.661,
  "input_tokens": 268539,
  "output_tokens": 6883
}
```

Cost ratio is not pre-computed for LLM gen; record `latency_seconds`, `input_tokens`, and `output_tokens` directly.

**Output JSON schema:**
```json
[
  {
    "question": "How many shares of common stock were outstanding ...",
    "question_slug": "how_many_shares_of_common_stock_were_outstanding_as_of_the_c",
    "latency_seconds": 187.661,
    "input_tokens": 268539,
    "output_tokens": 6883
  }
]
```

### Agent gen rules

Source files: `results/e2e_agent/rule_gen/{question_slug}_rule_gen.json` (one per question).

Relevant fields per file:
```json
{
  "question": "...",
  "question_slug": "...",
  "latency_seconds": 210.4,
  "agent_input_tokens": 142000,
  "agent_output_tokens": 9500,
  "judge_input_tokens": 18000,
  "judge_output_tokens": 600,
  "total_input_tokens": 160000,
  "total_output_tokens": 10100,
  "avg_cost_ratio": 0.029318,
  "merge_accuracy": 0.9
}
```

**Output JSON schema:**
```json
[
  {
    "question": "...",
    "question_slug": "...",
    "latency_seconds": 210.4,
    "total_input_tokens": 160000,
    "total_output_tokens": 10100,
    "avg_cost_ratio": 0.029318,
    "merge_accuracy": 0.9
  }
]
```

> Note: `avg_cost_ratio` here is the cost ratio measured on the sampled training docs during rule generation (the agent verifies rules against those 10 docs). `merge_accuracy` is the fraction of sampled docs correctly answered using the union of generated rules.

---

## Summary of Output File Conventions

Suggested output filenames for generated tables:

| Table type | Dataset | Rule type | Suggested output filename |
|---|---|---|---|
| performance | single cluster doc | llm-gen rules | `results/tables/perf_single_llm.json` |
| performance | single cluster doc | agent gen rules | `results/tables/perf_single_agent.json` |
| performance | mix doc case | llm-gen rules | `results/tables/perf_mix_llm.json` |
| performance | mix doc case | agent gen rules | `results/tables/perf_mix_agent.json` |
| rule gen status | single cluster doc | llm-gen rules | `results/tables/rulegen_single_llm.json` |
| rule gen status | single cluster doc | agent gen rules | `results/tables/rulegen_single_agent.json` |
