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
| 1 | **Agentic Claude QA** | gpt54 | — | — | — | — | Direct one-shot call, full doc in context |
| 2 | **Codex-style gpt54 QA** | gpt54 | — | — | — | — | gpt54 agentic loop with doc tools |

`cost` = mean over docs of `input_tokens / total_doc_tokens` (retrieval proxy).

---

## Strategy 1 — Agentic Claude QA (`src/baseline/agentic_claude_qa.py`)

### Approach

Given a question and a reconstructed document JSON, spawn a `claude -p` (Claude Code) session.
The agent uses all default tools (Read, Bash, etc.) to inspect the document and answer the question.
No rule pool, no span retrieval — the agent works directly from the raw document.

**Model options:**
- `opus47` — Claude Opus 4.7 (`claude-opus-4-7`) as the agent brain
- `gpt54` — GPT-4.5 (Azure) via a single direct chat-completion call (non-agentic)

### Metrics logged per (question, doc) pair

| Field | Description |
|-------|-------------|
| `answer` | The model's answer string |
| `input_tokens` | Total input tokens consumed (agent outer loop) |
| `output_tokens` | Total output tokens consumed |
| `latency_seconds` | Wall-clock time from call to answer |
| `total_cost_usd` | Reported by claude CLI (opus47 only) |
| `model` | Model identifier used |
| `status` | `ok`, `timeout`, `exit_N`, or `error` |

### Usage

```bash
# Run all questions × all sampled docs
python src/baseline/run_eval.py --baseline agentic_claude_qa --model opus47 --split sampled

# Run all questions × all unsampled docs
python src/baseline/run_eval.py --baseline agentic_claude_qa --model gpt54 --split unsampled

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

## Strategy 2 — Codex-style gpt54 QA (`src/baseline/codex_gpt54_qa.py`)

### Approach

Mirrors the Codex CLI pattern: gpt54 drives an **inner agentic loop** with three document tools.
Claude Code is the outer orchestrator (picks question/doc, writes results).
All reasoning and iteration decisions are made by gpt54 inside the loop — Claude never sees intermediate steps.

**Architecture:**
```
Claude Code (outer)
  └── run_eval.py iterates questions × docs
        └── codex_gpt54_qa.run_qa()
              └── gpt54 agentic loop (inner)
                    ├── tool: get_doc_info   — total pages + span count
                    ├── tool: read_page      — all spans on a page
                    └── tool: search_spans   — keyword search across doc
                  iterates until no more tool calls → returns final answer
```

**Tools available to gpt54:**

| Tool | Description |
|------|-------------|
| `get_doc_info` | Returns total page count and span count |
| `read_page(page_no)` | Returns all spans on a given page (text, bold, size, label) |
| `search_spans(keyword)` | Case-insensitive keyword search, max 30 results |

**Key difference from Strategy 1:**
- Strategy 1 (opus47): Claude agent reads the full doc in one pass via Read tool
- Strategy 2 (gpt54): gpt54 reads targeted sections via tool calls — more like how Codex CLI works

### Metrics logged (per-doc JSON)

Same schema as Strategy 1, plus:

| Field | Description |
|-------|-------------|
| `iterations` | Number of LLM turns in the inner loop |
| `tool_calls` | Total tool calls made across all iterations |

### Usage

```bash
# Single pair
python src/baseline/codex_gpt54_qa.py \
    --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
    --question "What is the registrant's telephone number?"

# All questions × sampled docs
python src/baseline/run_eval.py --baseline codex_gpt54_qa --model gpt54 --split sampled
```

### Output location

```
baseline_results/financebench/codex_gpt54_qa_gpt54/
    <question_slug>/<doc_name>.json
    summary.json
```

---

## How to add a new baseline strategy

1. Create `src/baseline/<strategy_name>.py` with a `run_qa(doc, question, **kwargs) -> dict` function.
2. Add a row to the summary table above.
3. Add a method-description section below.
4. Run `src/baseline/run_eval.py --baseline <name>` to populate results.
