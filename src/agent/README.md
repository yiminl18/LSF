# Agent Subsystem

This document describes the current `src/agent` architecture, entry points, and common experiment commands. All commands assume they are run from the repository root, `/path/to/the/LSF`, with `PYTHONPATH=src`. The main config is `src/agent/config_pdfs_10doc.yaml`, and its default dataset root is `datasets/pdfs/latest`. 

## Architecture Overview

`agent.rules` is the rule layer. It owns the `RangeRule` schema, JSON parsing, rule execution, and answer scoring. The core primitive is rule apply: apply one rule to one document and retrieve one or more candidate spans. Phase A (generate rules on sampled docs), Phase B (evaluate rules on unsampled docs), per-rule evaluation, union evaluation, and cascade evaluation all use this same retrieval primitive.

`agent.rule_runtime` is the shared runtime. It handles query/doc/label loading, `best_rules.json` artifacts, holdout evaluation, and the deployable cascade policy. Cascade is not another rule-apply mechanism. It is a multi-rule deployment policy: rank frozen rules by sampled accuracy, try them one by one on each holdout document, stop when generation returns an answer other than `information not found.`, then judge once.

`agent.reflection_agent` is the bundle-based rule generation path. It packages sampled documents into model prompts and asks the model to generate rules. It currently supports the `full_bundle_reference` and `grouped_433` packaging modes.

`agent.tool_agent` is the tool-calling agent path. Its CLI runs Phase A and Phase B. Phase A explores, validates, and selects rules on sampled documents. Phase B evaluates frozen rules on holdout documents and runs per-rule, union, and cascade metrics by default. Supported Phase A modes are `single_shot`, `diverse`, `hybrid`, and `curriculum`. The historical experiment name `tool-agent-trivial` maps to `--mode single_shot`.

Prompt files are split by strategy: bundle/reflection prompts live under `prompts/reflection_agent`, while tool-agent prompts live under `prompts/tool_agent`.

## Bundle Baseline: Full Bundle / Grouped

Bundle Phase A uses `agent.reflection_agent.runner.run_baseline_sweep`. It writes artifacts such as `best_rules.json`, `summary.json`, and `eval_rows.jsonl` under `output_root/q{query_idx}/{packaging_mode}/`.

```bash
cd /Users/chiyuh/Workspace/LSF
PYTHONPATH=src python - <<'PY'
from agent.reflection_agent.runner import run_baseline_sweep

run_baseline_sweep(
    packaging_modes=("full_bundle_reference", "grouped_433"),
    query_indices=(3,),
    config_path="src/agent/config_pdfs_10doc.yaml",
    output_root="output/agent/financial_baseline_runner",
    llm_provider="azure",
    llm_model="gpt-5.4-mini",
)
PY
```

The standard bundle Phase B holdout entry point is `agent.rule_runtime.holdout`. It evaluates per-rule and union behavior by default. `--packaging-mode` points to the packaging mode already produced by Phase A, such as `grouped_433` or `full_bundle_reference`.

```bash
PYTHONPATH=src python -m agent.rule_runtime.holdout \
  --config src/agent/config_pdfs_10doc.yaml \
  --packaging-mode grouped_433 \
  --queries 3 \
  --max-docs 25 \
  --llm-provider azure \
  --llm-model gpt-5.4-mini \
  --output-root output/agent/financial_baseline_runner
```

To compare a bundle baseline against the tool-agent Phase B deployable cascade metric, run `agent.rule_runtime.deploy` as an additional step. It reads an existing `best_rules.json` and runs the same cascade policy on holdout documents.

```bash
PYTHONPATH=src python -m agent.rule_runtime.deploy \
  --query-idx 3 \
  --config src/agent/config_pdfs_10doc.yaml \
  --in-best-rules output/agent/financial_baseline_runner/q3/grouped_433/best_rules.json \
  --output-dir output/agent/financial_baseline_runner/q3/grouped_433/deploy \
  --max-holdout-docs 25 \
  --llm-provider azure \
  --llm-model gpt-5.4-mini
```

## Tool-Agent Phase A

The unified tool-agent entry point is `agent.tool_agent.cli`. Phase A writes to `output_root/q{query_idx}/{experiment_name}/phase_a/`. Key artifacts include `best_rules.json`, `phase_a_docs.json`, `phase_a_report.json`, `phase_a_report.md`, and `trajectory.jsonl`.

`tool-agent-trivial` uses `single_shot` mode:

```bash
cd /Users/chiyuh/Workspace/LSF
PYTHONPATH=src python -m agent.tool_agent.cli \
  --config src/agent/config_pdfs_10doc.yaml \
  --queries 3 \
  --phase a \
  --max-docs 10 \
  --agent-provider azure \
  --agent-model gpt-5.4-mini \
  --max-turns 15 \
  --budget 2.0 \
  --output-root output/agent/tool_agent \
  --experiment-name tool-agent-trivial \
  --mode single_shot
```

Other Phase A variants only need different experiment names and modes:

```bash
# diverse: append-only diverse rule generation
--experiment-name tool-agent-diverse --mode diverse

# hybrid: multipath diverse exploration + union dedup + set-cover selection
--experiment-name tool-agent-hybrid --mode hybrid --multipath-n 2 --partition-seed 42

# curriculum: turn-aware broad-to-narrow diverse generation
--experiment-name tool-agent-curriculum --mode curriculum
```

## Tool-Agent Phase B

Tool-agent Phase B reads `output_root/q{query_idx}/{experiment_name}/phase_a/best_rules.json`. It prefers the sampled exclusion set in `phase_a_docs.json` when selecting holdout documents. Phase B writes to `output_root/q{query_idx}/{experiment_name}/phase_b/`. Key artifacts include `holdout_report.json`, `holdout_report.md`, `holdout_per_rule_rows.jsonl`, `holdout_union_rows.jsonl`, and `holdout_cascade_rows.jsonl`.

```bash
cd /Users/chiyuh/Workspace/LSF
PYTHONPATH=src python -m agent.tool_agent.cli \
  --config src/agent/config_pdfs_10doc.yaml \
  --queries 3 \
  --phase b \
  --max-docs 10 \
  --max-holdout-docs 25 \
  --holdout-strategy random \
  --agent-provider azure \
  --agent-model gpt-5.4-mini \
  --eval-provider azure \
  --eval-model gpt-5.4-mini \
  --output-root output/agent/tool_agent \
  --experiment-name tool-agent-trivial \
  --mode single_shot
```

The same CLI can run Phase A and Phase B back to back with `--phase both`:

```bash
PYTHONPATH=src python -m agent.tool_agent.cli \
  --config src/agent/config_pdfs_10doc.yaml \
  --queries 3 \
  --phase both \
  --max-docs 10 \
  --max-holdout-docs 25 \
  --holdout-strategy random \
  --agent-provider azure \
  --agent-model gpt-5.4-mini \
  --eval-provider azure \
  --eval-model gpt-5.4-mini \
  --max-turns 15 \
  --budget 2.0 \
  --output-root output/agent/tool_agent \
  --experiment-name tool-agent-trivial \
  --mode single_shot
```

## Common Checks

Use dry-run to check tool-agent command shape and sampled/holdout document counts before making real LLM calls.

```bash
PYTHONPATH=src python -m agent.tool_agent.cli \
  --config src/agent/config_pdfs_10doc.yaml \
  --queries 3 \
  --phase both \
  --max-docs 10 \
  --max-holdout-docs 25 \
  --agent-provider azure \
  --agent-model gpt-5.4-mini \
  --output-root output/agent/tool_agent \
  --experiment-name tool-agent-trivial \
  --mode single_shot \
  --dry-run
```
