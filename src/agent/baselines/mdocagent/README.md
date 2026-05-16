# MDocAgent Baseline

Multi-modal multi-agent baseline (arXiv:2503.13964). Five agents: general,
critical, text, image, summarizing.

## Prerequisites

### 1. Initialise the submodule

```bash
git submodule update --init src/agent/baselines/mdocagent/upstream
```

### 2. Run install.sh (one-time, modifies the upstream venv)

```bash
cd src/agent/baselines/mdocagent/upstream/MDocAgent
bash install.sh
cd -
```

### 3. Set environment variables

The wrapper generates a runtime Hydra model config that points upstream
MDocAgent at the LSF OpenAI-compatible adapter. For OpenRouter, set:

```bash
export OPENROUTER_API_KEY=your-openrouter-key
```

For Azure GPT-5.4-family models, the wrapper maps the existing `AZURE_54_*` or
`AZURE_54MINI_*` variables into the subprocess environment and uses
`max_completion_tokens`.

## Running

### Prepare inputs for one document

```bash
PYTHONPATH=src python -m agent.baselines.mdocagent.adapter \
    --config src/agent/config_pdfs_10doc.yaml \
    --query 0 --doc-id AMAZON_2015_10K
```

### Run the full baseline sweep (one query, one doc — smoke test)

```bash
PYTHONPATH=src python -m agent.run_pipeline \
    --experiment baseline-mdocagent \
    --queries 1 \
    --max-docs 1 \
    --llm-provider openrouter \
    --llm-model openai/gpt-4o
```

### Full sweep

```bash
PYTHONPATH=src python -m agent.run_pipeline \
    --experiment baseline-mdocagent \
    --llm-provider openrouter \
    --llm-model openai/gpt-4o
```

## Where results land

- Per-call subprocess log: `.cache/mdocagent/logs/<run-name>.log`
- MDocAgent result JSON: `src/agent/baselines/mdocagent/upstream/MDocAgent/results/lsf/<run-name>/<YYYY-MM-DD-HH-MM>.json`
- Baseline rows: `output/agent/baselines/<dataset>/mdocagent/q<idx>/baseline_rows.jsonl`
- Summary: `output/agent/baselines/<dataset>/mdocagent/q<idx>/baseline_summary.json`

## Retrieval deviation

The paper uses ColBERT for page retrieval. We bypass this by pre-supplying
the first 10 page indices as both text and image retrieved pages
(`text-top-10-question`, `image-top-10-question`). This avoids the ColBERT
dependency and prevents context overflow on 10-K filings (80-200 pages).
The 5-agent reasoning pipeline is otherwise unchanged.

## Running unit tests (no real subprocess)

```bash
PYTHONPATH=src python3 -m pytest test/test_mdocagent.py -v
```
