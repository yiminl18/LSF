#!/usr/bin/env bash
# Kill any running grid, then launch the agent_codex-ONLY financebench grid (6 combos).
# llm_coarse is dropped (times out on financebench's large docs).
cd ~/LSF
export PATH=$HOME/.npm-global/bin:$PATH
export AZURE_OPENAI_API_KEY=$(awk -F': ' '/^api_key:/{print $2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt)
pkill -9 -f run_grid_parallel.py 2>/dev/null
pkill -9 -f 'src/pipeline.py' 2>/dev/null
sleep 3
nohup python3 scripts/run_grid_parallel.py \
  --dataset financebench --cluster multi_cluster \
  --queries data/financebench/multi_cluster_queries.json \
  --output results/financebench/grid --rules rules/financebench/grid \
  --proc-dir data/financebench/processing \
  --rule-gens agent_codex_gpt54 --jobs 2 \
  > logs/fb_grid_agentcodex.log 2>&1 &
echo "agentcodex grid launched pid=$!"
