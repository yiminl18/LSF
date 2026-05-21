# Baselines

Each file `agentic_<name>.py` exports a `run_qa(doc_path, question, *, model, timeout, log_dir, log_stem, **kwargs) -> dict` and a CLI `main()`. `run_eval.py` is the batch driver that sweeps a baseline across a dataset, judges every answer with gpt-5.4, and writes per-doc JSON + a `summary.json`.

## Available baselines

| Module | Reads | Notes |
| --- | --- | --- |
| `agentic_codex_qa` | reconstructed JSON | Codex CLI agent |
| `agentic_claude_qa` | reconstructed JSON | Claude API agent |
| `agentic_mdocagent` | PDF | Multi-modal multi-agent ([arXiv:2503.13964](https://arxiv.org/abs/2503.13964)), runs the upstream submodule in `mdocagent/upstream/MDocAgent/` as a subprocess |

`run_eval.py` auto-discovers a baseline via `--baseline <module_name>`. PDF-only baselines must export `SUPPORTS_PDF_INPUT = True`.

## Prerequisites

Credentials live at `local/azure.json` (gpt-5.4 inline + optional `key_file_cheap` pointer for gpt-5.4-mini). Both files are gitignored.

For MDocAgent: `git submodule update --init src/baseline/mdocagent/upstream/MDocAgent` and run a one-time install (see upstream `install.sh`).

## Quick start

**Single (doc, question) smoke** — no judge, prints the raw `run_qa` dict:

```bash
$ENV src/baseline/agentic_mdocagent.py \
    --doc data/nopv/raw/<doc>.pdf \
    --question "On what date was this Notice issued?" \
    --model gpt54mini
```

**Batch sweep** — every (doc, question) pair, judged, persisted:

```bash
$ENV src/baseline/run_eval.py \
    --baseline agentic_mdocagent \
    --model gpt54mini \
    --dataset nopv \
    --max-docs 2
```

Common flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--baseline NAME` | module under `baseline/` to invoke | `agentic_claude_qa` |
| `--model ALIAS` | `gpt54`, `gpt54mini`, `opus47` (baseline-specific) | `opus47` |
| `--dataset NAME` | `financebench` or `nopv` | `financebench` |
| `--split` | `sampled` / `unsampled` (financebench only) | `sampled` |
| `--max-docs N` | alphabetically first N docs not yet completed | all |
| `--question-slug PREFIX` | run only questions whose slug starts with PREFIX | all |
| `--timeout SEC` | per (doc, q) subprocess cap | 300 |
| `--no-skip-existing` | force re-run docs that have a saved JSON | off |
| `--output-name DIR` | override the `<baseline>_<model>` output subdir | — |

The judge is hard-wired to gpt-5.4 for cross-model comparability; only the gen model switches via `--model`.

## Output layout

```
baseline_results/<dataset>/<baseline>_<model>/
├── run_metadata.json                # config for the batch (when --max-docs is set)
├── summary.json                     # per-question aggregates
└── <question_slug>/
    ├── <doc_name>.json              # one (doc, q) record (answer, correct, all telemetry)
    └── logs/
        ├── <doc_name>.mdocagent.log         # subprocess stdout/stderr (mdocagent only)
        └── <doc_name>.mdocagent.jsonl       # per-agent-call token/cost (mdocagent only)
```

Each per-doc record splits gen / judge / total cost & latency:

```json
{
  "answer": "...", "correct": true, "model": "gpt-5.4-mini", "judge_model": "gpt-5.4",
  "gen_input_tokens": ..., "gen_output_tokens": ..., "gen_latency_seconds": ..., "gen_cost_usd": ...,
  "judge_input_tokens": ..., "judge_output_tokens": ..., "judge_latency_seconds": ..., "judge_cost_usd": ...,
  "total_cost_usd": ..., "total_latency_seconds": ...
}
```

`summary.json` aggregates the same fields as `avg_gen_*`, `avg_judge_*`, `avg_total_*` plus `accuracy` and `n_correct / n`.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Azure deployment name missing` | mini key file lacks `deployment:` | add `deployment: gpt-5.4-mini` to `local/azure_gpt54mini.txt` |
| `does not declare SUPPORTS_PDF_INPUT` | tried codex/claude on nopv | nopv is PDF-only — use `agentic_mdocagent` |
| `MDocAgent upstream not initialised` | submodule not cloned | `git submodule update --init src/baseline/mdocagent/upstream/MDocAgent` |
| `nopv labels not found at all_labels.json` | GT not generated | run `data/nopv/generate_labels.py [--max-docs N]` |
| subprocess `timeout` status | default 300s too tight | pass `--timeout 900` |
| leftover `src/baseline/mdocagent/upstream/MDocAgent/data/run-*` | a previous mdocagent run failed/timed out (kept for debugging) | `rm -rf` them when done debugging |

## Reference cost (gpt-5.4-mini × nopv)

| Scope | Time | Cost |
| --- | --- | --- |
| 1 (doc, q) | ~15s | ~$0.024 |
| 1 doc × 12 q | ~3 min | ~$0.30 |
| 2 doc × 12 q | ~6 min | ~$0.60 |
| 242 doc × 12 q (full) | ~12 h | ~$70 |

gpt-5.4 is roughly 3–4× more expensive at similar latency.
