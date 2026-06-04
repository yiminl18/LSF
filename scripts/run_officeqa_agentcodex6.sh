#!/usr/bin/env bash
# After the current officeqa run finishes, run the 6 agent_codex full pipelines
# on officeqa (llm_coarse is stopped for officeqa due to context overflow).
#   {random, fps} x agent_codex_gpt54 x {agentic_codex_gpt54, p_mini, p_hybrid}
# Uses the DAG driver restricted to agent_codex (--rule-gens), so precompute runs
# once per (sampling, rule_gen) and the 3 refiners share it (no race), with
# --skip-existing reusing the rule pools + the 2 agentic_codex combos already done.
# Launch detached (setsid) so it survives the laptop being off.

cd "$(dirname "$0")/.."

echo "[ac6] $(date '+%F %T') waiting for current officeqa run to finish ..."
while pgrep -f run_officeqa_top4.sh >/dev/null; do sleep 60; done
while ps -eo comm,args | awk '$1=="python3"' | grep -q "dataset officeqa"; do sleep 60; done
echo "[ac6] $(date '+%F %T') current run done; deploying fix + launching 6 agent_codex combos"

git pull --ff-only origin yiming-dev 2>&1 | tail -2

exec python3 scripts/run_grid_parallel.py \
  --dataset officeqa --cluster all_docs \
  --queries data/officeqa/queries.json \
  --output results/officeqa/grid \
  --rules rules/officeqa/grid \
  --proc-dir data/officeqa/normalized_json \
  --rule-gens agent_codex_gpt54 \
  --jobs 4
