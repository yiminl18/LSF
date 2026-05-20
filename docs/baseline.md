# Baseline Strategies — Version Tracking

This document indexes every baseline QA strategy implemented under `src/baseline/`.
Each baseline takes a **question + document** and produces an answer, logging tokens and latency.
Baselines are the comparison floor for the LSF rule-based retrieval pipeline.

**Dataset:** FinanceBench single-cluster (10 sampled docs, 50 unsampled docs, 10 questions).
**Evaluation:** gpt54 judge (same as rule-gen eval).

---

## Summary table

| # | Strategy | Model | sAcc | cost_s | uAcc | cost_u | Notes |
|---|----------|-------|-----:|-------:|-----:|-------:|-------|
| 1 | **Agentic Claude QA** | opus47 | 0.920 | 1.3484 | — | — | Claude agent reads full doc with tools |
| 2 | **Agentic Codex QA** | gpt54 | 0.940 | 1.1807 | — | — | Codex agent reads full doc with default tools |

`cost` = mean over docs of `input_tokens / total_doc_tokens` (retrieval proxy).
For Agentic Codex QA, `input_tokens` includes cached input tokens for parity with Claude.

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

**Additional 10-doc single-cluster unsampled batch**
(`baseline_results/financebench/agentic_codex_qa_gpt54_single_cluster_extra`):
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

```
baseline_results/
└── financebench/
    └── agentic_codex_qa_gpt54/
        ├── <question_slug>/
        │   ├── <doc_name>.json
        │   ├── logs/
        │   │   ├── <doc_name>.codex.jsonl
        │   │   └── <doc_name>.codex.last.txt
        │   └── ...
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
  "codex_log_path": "baseline_results/financebench/agentic_codex_qa_gpt54/.../logs/JPMORGAN_2023_10K.codex.jsonl",
  "codex_last_message_path": "baseline_results/financebench/agentic_codex_qa_gpt54/.../logs/JPMORGAN_2023_10K.codex.last.txt"
}
```

---

## How to add a new baseline strategy

1. Create `src/baseline/<strategy_name>.py` with a `run_qa(doc, question, **kwargs) -> dict` function.
2. Add a row to the summary table above.
3. Add a method-description section below.
4. Run `src/baseline/run_eval.py --baseline <name>` to populate results.
