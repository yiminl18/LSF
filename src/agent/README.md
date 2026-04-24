# Agent Subsystem

This document describes the current `src/agent` architecture and the focused
experiment runner. Run commands from the repository root with `PYTHONPATH=src`.
The default config is `src/agent/config_pdfs_10doc.yaml`, whose dataset root is
`datasets/pdfs/latest`.

## Quick Start

Use `agent.run_pipeline` for routine experiments. It runs one explicit
experiment at a time.

```bash
cd /Users/chiyuh/Workspace/LSF

PYTHONPATH=src python -m agent.run_pipeline \
  --experiment tool-agent-trivial \
  --phase both \
  --queries 3 \
  --max-holdout-docs 25
```

Run the grouped bundle baseline:

```bash
PYTHONPATH=src python -m agent.run_pipeline \
  --experiment bundle-grouped \
  --phase both \
  --queries 3 \
  --max-holdout-docs 25
```

Run bundle holdout and then the optional cascade deploy comparison:

```bash
PYTHONPATH=src python -m agent.run_pipeline \
  --experiment bundle-grouped \
  --phase b \
  --queries 3 \
  --max-holdout-docs 25 \
  --bundle-deploy
```

Run the hybrid tool-agent Phase A path:

```bash
PYTHONPATH=src python -m agent.run_pipeline \
  --experiment tool-agent-hybrid \
  --phase a \
  --queries 3 \
  --max-holdout-docs 25 \
  --multipath-n 2
```

Check command shape without LLM calls:

```bash
PYTHONPATH=src python -m agent.run_pipeline \
  --experiment tool-agent-trivial \
  --phase both \
  --queries 3 \
  --max-holdout-docs 25 \
  --dry-run
```

Supported experiments:

- `bundle-full`: bundle baseline with `full_bundle_reference`
- `bundle-grouped`: bundle baseline with `grouped_433`
- `tool-agent-trivial`: tool-agent `single_shot`
- `tool-agent-diverse`: tool-agent `diverse`
- `tool-agent-hybrid`: tool-agent `hybrid`
- `tool-agent-curriculum`: tool-agent `curriculum`

Common defaults:

- `--config src/agent/config_pdfs_10doc.yaml`
- `--queries 3`
- `--phase both`
- `--agent-provider azure --agent-model gpt-5.4-mini`
- `--max-docs 10 --max-holdout-docs 25 --holdout-seed 42`
- bundle outputs: `output/agent/financial_baseline_runner`
- tool-agent outputs: `output/agent/tool_agent`

## Architecture Overview

`agent.rules` is the rule layer. It owns the `RangeRule` schema, JSON parsing,
rule execution, and answer scoring. The core primitive is rule apply: apply one
rule to one document and retrieve one or more candidate spans. Phase A, Phase B,
per-rule evaluation, union evaluation, and cascade evaluation all use this same
retrieval primitive.

`agent.rule_runtime` is the shared runtime. It handles query/doc/label loading,
`best_rules.json` artifacts, holdout evaluation, and the deployable cascade
policy. Cascade is not another rule-apply mechanism. It is a multi-rule
deployment policy: rank frozen rules by sampled accuracy, try them one by one
on each holdout document, stop when generation returns an answer other than
`information not found.`, then judge once.

`agent.reflection_agent` is the bundle-based rule generation path. It packages
sampled documents into model prompts and asks the model to generate rules. It
currently supports `full_bundle_reference` and `grouped_433`.

`agent.tool_agent` is the tool-calling agent path. Phase A explores, validates,
and selects rules on sampled documents. Phase B evaluates frozen rules on
holdout documents and runs per-rule, union, and cascade metrics by default.

Prompt files are split by strategy: bundle/reflection prompts live under
`prompts/reflection_agent`, while tool-agent prompts live under
`prompts/tool_agent`.

## Outputs

Bundle outputs are written under:

```text
output/agent/financial_baseline_runner/q{query_idx}/{packaging_mode}/
```

Tool-agent outputs are written under:

```text
output/agent/tool_agent/q{query_idx}/{experiment_name}/
```

Key Phase A artifacts include `best_rules.json`, sampled-doc metadata,
evaluation rows, summaries, and prompt/trajectory files depending on the
strategy. Tool-agent Phase B writes `holdout_report.json`,
`holdout_union_rows.jsonl`, and `holdout_cascade_rows.jsonl`.

## Lower-Level Entrypoints

Use these only for debugging a specific implementation path:

- `python -m agent.tool_agent.cli`
- `python -m agent.rule_runtime.holdout`
- `python -m agent.rule_runtime.deploy`
- `agent.reflection_agent.runner.run_baseline_sweep(...)`
