#!/usr/bin/env bash
# Single-combo officeqa run: fps / llm_coarse_gpt54 / p_hybrid (full chain to apply).
# llm_coarse was previously stopped for officeqa (JSON rule-gen prompt overflows the
# context window on the huge office docs). fps sampling reads almost nothing, so this
# is the first llm_coarse combo to retry. Runs the whole pipeline
# (sampling -> rule_gen -> precompute -> refine -> apply) in one pipeline.py call;
# --skip-existing reuses the staged fps sampling split and anything already on disk.
cd ~/LSF
export PATH=$HOME/.npm-global/bin:$PATH
export AZURE_OPENAI_API_KEY=$(awk -F': ' '/^api_key:/{print $2; exit}' ~/api_keys/azure_cloudbank/gpt-54_1.txt)
pkill -9 -f 'src/pipeline.py' 2>/dev/null
sleep 3
mkdir -p logs
nohup python3 src/pipeline.py \
  --sampling-strategy fps \
  --rule-gen-strategy  llm_coarse_gpt54 \
  --refine-strategy    p_hybrid \
  --apply-strategy     default \
  --queries-file       data/officeqa/queries.json \
  --dataset            officeqa \
  --cluster            all_docs \
  --output-dir         results/officeqa/grid \
  --processing-dir     data/officeqa/normalized_json \
  --stop-after         apply \
  --skip-existing \
  > logs/officeqa_fps_llmcoarse_phybrid.log 2>&1 &
echo "officeqa fps/llm_coarse/p_hybrid launched pid=$!"
