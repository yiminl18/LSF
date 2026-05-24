#!/bin/bash
# Sequential launcher for the agentic-gen pipeline on lsf.
# Tasks 1 (random sample) and 2 (FPS sample) — generate, then eval on sampled and unsampled.
set -uo pipefail
cd "$HOME/LSF"
export PATH="$HOME/.npm-global/bin:$PATH"
mkdir -p logs

phase() { echo "===== $(date '+%Y-%m-%d %H:%M:%S')  $1 ====="; }

phase "Task 1 generation — random sample (10 Q)"
python3 agent/run_agent_gen.py --sample-set random --budget 30 2>&1 | tee logs/task1_gen.log
echo "task1_gen exit=$?"

phase "Task 1 eval — sampled (10 docs)"
python3 test/run_eval_merge_agentic.py --sample-set random --split sampled 2>&1 | tee logs/task1_eval_sampled.log
echo "task1_eval_sampled exit=$?"

phase "Task 1 eval — unsampled (50 docs)"
python3 test/run_eval_merge_agentic.py --sample-set random --split unsampled 2>&1 | tee logs/task1_eval_unsampled.log
echo "task1_eval_unsampled exit=$?"

phase "Task 2 generation — FPS sample (10 Q)"
python3 agent/run_agent_gen.py --sample-set fps --budget 30 2>&1 | tee logs/task2_gen.log
echo "task2_gen exit=$?"

phase "Task 2 eval — sampled (10 docs)"
python3 test/run_eval_merge_agentic.py --sample-set fps --split sampled 2>&1 | tee logs/task2_eval_sampled.log
echo "task2_eval_sampled exit=$?"

phase "Task 2 eval — unsampled (49 docs)"
python3 test/run_eval_merge_agentic.py --sample-set fps --split unsampled 2>&1 | tee logs/task2_eval_unsampled.log
echo "task2_eval_unsampled exit=$?"

phase "ALL DONE"
