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
| 1 | **Agentic Claude QA** | opus47 | — | — | — | — | Not yet evaluated |
| 1 | **Agentic Claude QA** | gpt54 | — | — | — | — | Not yet evaluated |

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
# Single (question, doc) pair — dry run
python src/baseline/agentic_claude_qa.py \
    --doc data/financebench/processing/JPMORGAN_2023_10K_reconstructed.json \
    --question "What is the registrant's telephone number?" \
    --model opus47

# Run all questions × all sampled docs
python src/baseline/run_baseline_eval.py --strategy agentic --split sampled --model opus47

# Run all questions × all unsampled docs
python src/baseline/run_baseline_eval.py --strategy agentic --split unsampled --model gpt54
```

### Output layout

```
results/financebench_single_cluster/baseline/
└── agentic_claude_qa/
    └── <model>/
        ├── eval_sampled/
        │   ├── <slug>_sampled.json    # per-question results
        │   └── summary.json
        └── eval_unsampled/
            ├── <slug>_unsampled.json
            └── summary.json
```

### Per-question JSON schema

```json
{
  "question": "...",
  "question_slug": "...",
  "split": "sampled",
  "model": "claude-opus-4-7",
  "n": 10,
  "n_correct": 8,
  "accuracy": 0.800,
  "avg_latency": 12.3,
  "avg_input_tokens": 4500,
  "avg_output_tokens": 120,
  "avg_cost_usd": 0.045,
  "per_doc": [
    {
      "doc_name": "JPMORGAN_2023_10K",
      "answer": "(212) 270-6000",
      "ground_truth": "(212) 270-6000",
      "correct": true,
      "input_tokens": 4821,
      "output_tokens": 98,
      "latency_seconds": 11.4,
      "total_cost_usd": 0.042
    }
  ]
}
```

---

## How to add a new baseline strategy

1. Create `src/baseline/<strategy_name>.py` with a `run_qa(doc, question, **kwargs) -> dict` function.
2. Add a row to the summary table above.
3. Add a method-description section below.
4. Run `src/baseline/run_baseline_eval.py --strategy <name>` to populate results.
