# Baseline Strategies — Version Tracking

This document indexes every baseline QA strategy implemented under `src/baseline/`.
Most baselines take a **question + document** and produce an answer, logging tokens and latency.
Some planned variants change the execution granularity while keeping the same final per-pair result format.
Baselines are the comparison floor for the LSF rule-based retrieval pipeline.

**Dataset:** FinanceBench single-cluster (10 sampled docs, 50 unsampled docs, 10 questions).
**Evaluation:** gpt54 judge (same as rule-gen eval).

---

## Summary table

| # | Strategy | Model | Acc | Latency | CostRatio |
|---|----------|-------|----:|--------:|----------:|
| 1 | **Agentic Claude QA** | opus47 | 0.9466 | 10.3s | 0.9112 |
| 2 | **Agentic Claude QA** | sonnet | 0.8788 | 41.3s | 1.8006 |
| 3 | **Agentic Codex QA** | gpt54 | 0.9102 | 20.3s | 1.3221 |
| 4 | **Agentic Codex QA** | gpt54mini | 0.8800 | 15.1s | 1.2878 |
| 5 | **Agentic Codex QA (All Docs + All Queries)** | gpt54 | — | — | — |

- `Acc` = fraction correct over all completed (question, doc) pairs; judge: gpt54
- `CostRatio` = mean(input_tokens / total_doc_tokens) per pair; total_doc_tokens approximated as chars÷4 from reconstructed JSON text spans
- Rows 2 and 4 are based on partial runs (in progress)

---

## Strategy 1 — Agentic Claude QA (`src/baseline/agentic_claude_qa.py`)

### Approach

Given a question and a reconstructed document JSON, spawn a `claude -p` (Claude Code) session.
The agent uses all default tools (Read, Bash, etc.) to inspect the document and answer the question.
No rule pool, no span retrieval — the agent works directly from the raw document.

**Model:**
- `opus47` — Claude Opus 4.7 (`claude-opus-4-7`) as the agent brain

### Metrics logged per (question, doc) pair

| Field | Description |
|-------|-------------|
| `answer` | The model's answer string |
| `input_tokens` | Total input tokens consumed (agent outer loop) |
| `output_tokens` | Total output tokens consumed |
| `latency_seconds` | Wall-clock time from call to answer |
| `total_cost_usd` | Reported by claude CLI |
| `model` | Model identifier used |
| `status` | `ok`, `timeout`, `exit_N`, or `error` |

### Usage

```bash
# Run all questions × all sampled docs
python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled

# Single question
python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled \
    --question-slug what_is_the_registrants_telephone_number
```

### Output layout

```
baseline_results/
└── <dataset>/                              e.g. financebench
    └── <baseline>_<model>/                 e.g. agentic_claude_qa_opus47
        ├── <question_slug>/                one folder per question
        │   ├── <doc_name>.json             one file per (question, doc) pair
        │   └── ...
        └── summary.json                    mean accuracy, tokens, latency across questions
```

**Example (FinanceBench, opus47):**

```
baseline_results/
└── financebench/
    └── agentic_claude_qa_opus47/
        ├── what_is_the_registrants_telephone_number/
        │   ├── JPMORGAN_2023_10K.json
        │   ├── APPLE_2022_10K.json
        │   └── ...                         (one file per doc in the split)
        ├── what_is_total_assets_at_yearend_from_the_audited_balance_she/
        │   └── ...
        └── summary.json
```

### Per-doc JSON schema (`<question_slug>/<doc_name>.json`)

One file per (question, document) pair — the atomic unit of results.

```json
{
  "doc_name":        "JPMORGAN_2023_10K",
  "question":        "What is the registrant's telephone number?",
  "question_slug":   "what_is_the_registrants_telephone_number",
  "split":           "sampled",
  "ground_truth":    "(212) 270-6000",
  "answer":          "(212) 270-6000",
  "correct":         true,
  "status":          "ok",
  "input_tokens":    4821,
  "output_tokens":   98,
  "latency_seconds": 11.4,
  "total_cost_usd":  0.042,
  "model":           "claude-opus-4-7"
}
```

### Summary JSON schema (`summary.json`)

One entry per question, aggregated over all docs in the split.

```json
[
  {
    "question":              "What is the registrant's telephone number?",
    "question_slug":         "what_is_the_registrants_telephone_number",
    "split":                 "sampled",
    "model":                 "opus47",
    "n":                     10,
    "n_correct":             9,
    "accuracy":              0.9,
    "avg_input_tokens":      4650.3,
    "avg_output_tokens":     102.1,
    "avg_latency_seconds":   12.8
  }
]
```

---

## Strategy 2 — Agentic Codex QA (`src/baseline/agentic_codex_qa.py`)

### Approach

Given a question and a reconstructed document JSON, spawn a non-interactive
`codex exec` session. The Codex agent runs from the repo root with its default
tool environment, reads/searches the raw reconstructed document, and answers the
question directly.

No rule pool, no span retrieval — this is the GPT-5.4/Codex analogue of
Strategy 1.

**Model:**
- `gpt54` — alias resolved by the wrapper to the Codex model id `gpt-5.4`

**Measured sampled results:**
- `sAcc = 0.940`
- `cost_s = 1.1807`
- `latency_s = 18.46s`

Checked-in sampled artifacts now live under
`baseline_results/financebench/agentic_codex_qa_gpt54/single_cluster/batch_0`.

**Additional 10-doc single-cluster random batch**
(`baseline_results/financebench/agentic_codex_qa_gpt54/single_cluster/batch_1`):
- `Acc = 0.800`
- `cost = 1.3271`
- `latency = 17.15s`

**Invocation:**

```bash
codex --ask-for-approval never exec \
  --json \
  --color never \
  --model gpt-5.4 \
  --cd <repo-root> \
  --sandbox danger-full-access \
  --output-last-message <path> \
  <prompt>
```

### Metrics logged per (question, doc) pair

The baseline logs the common fields from Strategy 1 plus Codex-specific metadata:

| Field | Description |
|-------|-------------|
| `answer` | The model's answer string |
| `input_tokens` | Total input tokens from Codex `turn.completed.usage`, including cached input tokens |
| `output_tokens` | Total output tokens from Codex `turn.completed.usage` |
| `cached_input_tokens` | Cached-input subset of `input_tokens` reported by Codex |
| `reasoning_output_tokens` | Reasoning tokens reported by Codex |
| `latency_seconds` | Wall-clock time from call to answer |
| `total_cost_usd` | Always `null` unless Codex CLI starts reporting cost |
| `model` | Resolved model id, normally `gpt-5.4` |
| `status` | `ok`, `timeout`, `exit_N`, or `error` |
| `codex_thread_id` | Codex session/thread id from JSONL events |
| `codex_event_count` | Count of parseable Codex JSONL events |
| `codex_error_message` | Error payload if Codex reports one |
| `codex_log_path` | Path to saved Codex JSONL event log |
| `codex_last_message_path` | Path to saved final Codex message |

### Usage

```bash
# Run all questions × all sampled docs
python src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54 --split sampled

# Single question
python src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54 --split sampled \
    --question-slug what_is_the_registrants_telephone_number

# Single document/question smoke test
python src/baseline/agentic_codex_qa.py \
    --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
    --question "What is the registrant's telephone number?"
```

### Output layout

Current checked-in Codex baseline artifacts are organized by single-cluster batch:

```
baseline_results/
└── financebench/
    └── agentic_codex_qa_gpt54/
        └── single_cluster/
            ├── batch_0/                    sampled 10-doc run
            │   ├── <question_slug>/
            │   │   ├── <doc_name>.json
            │   │   ├── logs/
            │   │   │   ├── <doc_name>.codex.jsonl
            │   │   │   └── <doc_name>.codex.last.txt
            │   │   └── ...
            │   └── summary.json
            └── batch_1/                    another 10-doc random run
                ├── <question_slug>/
                │   ├── <doc_name>.json
                │   ├── logs/
                │   │   ├── <doc_name>.codex.jsonl
                │   │   └── <doc_name>.codex.last.txt
                │   └── ...
                ├── run_metadata.json
                └── summary.json
```

### Per-doc JSON schema (`<question_slug>/<doc_name>.json`)

```json
{
  "doc_name": "JPMORGAN_2023_10K",
  "question": "What is the registrant's telephone number?",
  "question_slug": "what_is_the_registrants_telephone_number",
  "split": "sampled",
  "ground_truth": "(212) 270-6000",
  "answer": "(212) 270-6000",
  "correct": true,
  "status": "ok",
  "input_tokens": 22466,
  "output_tokens": 311,
  "cached_input_tokens": 10624,
  "reasoning_output_tokens": 108,
  "latency_seconds": 18.6,
  "total_cost_usd": null,
  "model": "gpt-5.4",
  "codex_thread_id": "019e...",
  "codex_event_count": 12,
  "codex_error_message": null,
  "codex_log_path": "baseline_results/financebench/agentic_codex_qa_gpt54/single_cluster/batch_0/.../logs/JPMORGAN_2023_10K.codex.jsonl",
  "codex_last_message_path": "baseline_results/financebench/agentic_codex_qa_gpt54/single_cluster/batch_0/.../logs/JPMORGAN_2023_10K.codex.last.txt"
}
```

---

## Strategy 3 — Agentic Codex QA All (`src/baseline/agentic_codex_qa_gpt54_all.py`)

### Approach

This baseline keeps the same end goal as `agentic_codex_qa_gpt54`: generate one answer
for every `(question, document)` pair in a dataset, judged the same way and written in
the same result format.

The difference is execution granularity:
- `agentic_codex_qa_gpt54`: one Codex agent call per `(question, document)` pair
- `agentic_codex_qa_gpt54_all`: one Codex agent call over the full dataset scope at once

For `agentic_codex_qa_gpt54_all`, the agent receives:
- all queries for the dataset
- all documents for the dataset, including both sampled and unsampled portions when applicable
- the dataset's `.txt` document files as the source documents
- the default Codex tool environment from the repo

The agent is then responsible for producing answers for all documents and all queries
within one baseline execution scope, but it is free to perform whatever internal loops,
iterations, searches, planning steps, and tool calls it wants in order to get there.
This is not intended to constrain the agent to a single pass or a single turn.

The checked-in artifacts should still be materialized back into the same external layout
as the existing Codex baseline.

### Document source

For this baseline, all datasets should use the plain-text document version as the agent
input source.

That means:
- use dataset `.txt` files rather than reconstructed JSON files
- keep the final output schema the same as the existing baseline result folders
- keep evaluation at the `(question, document)` level after the global run completes

### Output contract

The output schema is intentionally unchanged from `agentic_codex_qa_gpt54` and the court
baseline layout under `baseline_results/court/agentic_codex_qa_gpt54/first_50`.

Expected layout:

```text
baseline_results/
└── <dataset>/
    └── agentic_codex_qa_gpt54_all/
        └── <split_or_scope>/
            ├── <question_slug>/
            │   ├── <doc_name>.json
            │   ├── logs/
            │   │   ├── <doc_name>.codex.jsonl
            │   │   └── <doc_name>.codex.last.txt
            │   └── ...
            ├── run_metadata.json
            └── summary.json
```

Each `<question_slug>/<doc_name>.json` should keep the same fields as Strategy 2:
- answer / correctness fields
- token / latency fields
- Codex trace metadata such as `codex_log_path` and `codex_last_message_path`

### Runner

This baseline uses a dedicated dataset-scope runner:
- `src/baseline/run_eval_all.py`

Example:

```bash
python src/baseline/run_eval_all.py \
  --baseline agentic_codex_qa_gpt54_all \
  --model gpt54 \
  --split all_docs
```

### Measurement requirements

This baseline must preserve the same final evaluation target as the existing per-pair
baseline, but it is especially important to measure the full agent process correctly.

Required accounting:
- precisely track total input tokens consumed across the whole process
- precisely track total output tokens consumed across the whole process
- precisely track reasoning / thinking tokens across the whole process when available
- precisely track end-to-end latency for the whole process

After the full run finishes, these totals should be propagated into the final checked-in
results so that:
- per-pair outputs remain compatible with the existing schema
- aggregate summaries can report accuracy, token usage, and latency for this baseline

The final reported metric of interest is still accuracy over all `(question, document)`
pairs, updated after the complete run is materialized into the standard result format.

### Intended use

This baseline is meant to measure whether a single long-running Codex agent with global
access to the dataset can outperform or behave differently from the per-pair agent setup,
without changing the evaluation target or downstream artifact format.

### Status

Implemented. Metrics will be filled in after baseline runs are completed.

---

## How to add a new baseline strategy

1. Create `src/baseline/<strategy_name>.py` with a `run_qa(doc, question, **kwargs) -> dict` function.
2. Add a row to the summary table above.
3. Add a method-description section below.
4. Run `src/baseline/run_eval.py --baseline <name>` to populate results.
