#!/usr/bin/env bash
# After the current officeqa top-4 run finishes, deploy the timeout+truncation
# fix and re-run the llm_coarse combos that need it:
#   - fps/llm_coarse/p_hybrid          : always (it hit 28 context_length errors)
#   - random/llm_coarse/agentic_codex  : only if its log shows context_length errors
# Reuses clean work via --skip-existing (rule pools, precompute, clean apply);
# only the degraded p_hybrid apply is cleared so it regenerates with truncation.
#
# Launch detached (setsid) so the whole chain survives the laptop being off.

cd "$(dirname "$0")/.."
TOPLOG=logs/officeqa_top4

echo "[fixrerun] $(date '+%F %T') waiting for current officeqa top-4 to finish ..."
while pgrep -f run_officeqa_top4.sh >/dev/null; do sleep 60; done
# also drain any lingering officeqa pipeline.py (python3 procs only -> no self-match)
while ps -eo comm,args | awk '$1=="python3"' | grep -q "dataset officeqa"; do sleep 60; done
echo "[fixrerun] $(date '+%F %T') current run finished; deploying fix + re-running"

git pull --ff-only origin yiming-dev 2>&1 | tail -2

export PATH="$HOME/.npm-global/bin:$PATH"
export AZURE_OPENAI_API_KEY="$(awk -F': ' '/^api_key:/{print $2; exit}' "$HOME/api_keys/azure_cloudbank/gpt-54_1.txt")"

# p_hybrid: clear its (context-degraded) apply so it re-runs with truncation
rm -rf results/officeqa/grid/apply/fps/llm_coarse_gpt54/p_hybrid
RERUN=("fps llm_coarse_gpt54 p_hybrid")

# random/llm_coarse/agentic: only re-run if it accrued context errors
if grep -q context_length_exceeded "$TOPLOG/random_llm_coarse_gpt54_agentic_codex_gpt54.log" 2>/dev/null; then
  echo "[fixrerun] random/llm_coarse/agentic had ctx-errors -> clearing its apply + re-running"
  rm -rf results/officeqa/grid/apply/random/llm_coarse_gpt54/agentic_codex_gpt54
  RERUN+=("random llm_coarse_gpt54 agentic_codex_gpt54")
else
  echo "[fixrerun] random/llm_coarse/agentic clean (no ctx-errors) -> leaving as-is"
fi

mkdir -p logs/officeqa_fixrerun
for combo in "${RERUN[@]}"; do
  read -r s g r <<< "$combo"
  echo "[fixrerun] re-running $s/$g/$r"
  python3 src/pipeline.py \
    --sampling-strategy "$s" --rule-gen-strategy "$g" --refine-strategy "$r" \
    --apply-strategy default --queries-file data/officeqa/queries.json \
    --dataset officeqa --cluster all_docs --processing-dir data/officeqa/normalized_json \
    --output-dir results/officeqa/grid --skip-existing \
    > "logs/officeqa_fixrerun/${s}_${g}_${r}.log" 2>&1 &
done
wait
echo "[fixrerun] $(date '+%F %T') done"
