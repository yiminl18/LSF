#!/usr/bin/env bash
# Run the 4 finance baselines on the 5 easy-but-not-baseline questions, 50 GT docs.
set -u
cd ~/LSF
export PATH=$HOME/.npm-global/bin:$PATH

QF=data/financebench/easy5_baseline_queries.txt
LS=data/financebench/sample/single_cluster/random/easy5_sample_doc_labels.json
LU=data/financebench/sample/single_cluster/random/easy5_unsampled_doc_labels.json
LA=data/financebench/sample/single_cluster/random/easy5_all_doc_labels.json
KFULL=$HOME/api_keys/azure_cloudbank/gpt-54_1.txt
KMINI=$HOME/api_keys/azure_cloudbank/gpt-54-mini.txt
getkey(){ awk -F': ' '/^api_key:/{print $2; exit}' "$1"; }

echo "===== [1/4] per-pair gpt54  $(date) ====="
export AZURE_OPENAI_API_KEY=$(getkey $KFULL)
python3 src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54 --split sampled   --queries-file $QF --labels-file $LS --output-name agentic_codex_qa_gpt54_easy5
python3 src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54 --split unsampled --queries-file $QF --labels-file $LU --output-name agentic_codex_qa_gpt54_easy5

echo "===== [2/4] per-pair gpt54mini  $(date) ====="
export AZURE_OPENAI_API_KEY=$(getkey $KMINI)
python3 src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54mini --split sampled   --queries-file $QF --labels-file $LS --output-name agentic_codex_qa_gpt54mini_easy5
python3 src/baseline/run_eval.py --baseline agentic_codex_qa --model gpt54mini --split unsampled --queries-file $QF --labels-file $LU --output-name agentic_codex_qa_gpt54mini_easy5

echo "===== [3/4] All gpt54  $(date) ====="
export AZURE_OPENAI_API_KEY=$(getkey $KFULL)
python3 src/baseline/run_eval_all.py --baseline agentic_codex_qa_gpt54_all --split all_docs --queries-file $QF --labels-file $LA --output-name agentic_codex_qa_gpt54_all_easy5

echo "===== [4/4] All gpt54mini  $(date) ====="
export AZURE_OPENAI_API_KEY=$(getkey $KMINI)
python3 src/baseline/run_eval_all.py --baseline agentic_codex_qa_gpt54mini_all --split all_docs --queries-file $QF --labels-file $LA --output-name agentic_codex_qa_gpt54mini_all_easy5

echo "===== DONE  $(date) ====="
