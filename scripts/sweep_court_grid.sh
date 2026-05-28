#!/usr/bin/env bash
# Sweep the 12-combo planned grid on court — all 13 questions, full corpus.
#
# Combos (refine: agentic-codex first, then p_mini, then p_hybrid; this order
# means agentic combos finish without precompute, and p_mini's precompute is
# reused by p_hybrid for the same (sampling, rule_gen) pair):
#
#   sampling  rule_gen           refine                apply
#   --------  -----------------  --------------------  -------
#   random    llm_coarse_gpt54   agentic_codex_gpt54   default
#   random    llm_coarse_gpt54   p_mini                default
#   random    llm_coarse_gpt54   p_hybrid              default
#   random    agent_codex_gpt54  agentic_codex_gpt54   default
#   random    agent_codex_gpt54  p_mini                default
#   random    agent_codex_gpt54  p_hybrid              default
#   fps       llm_coarse_gpt54   agentic_codex_gpt54   default
#   fps       llm_coarse_gpt54   p_mini                default
#   fps       llm_coarse_gpt54   p_hybrid              default
#   fps       agent_codex_gpt54  agentic_codex_gpt54   default
#   fps       agent_codex_gpt54  p_mini                default
#   fps       agent_codex_gpt54  p_hybrid              default
#
# Idempotent: each pipeline.py invocation uses --skip-existing, so re-runs only
# do the work missing from disk. Safe to interrupt and resume.

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="court"
CLUSTER="all_docs"
QUERIES="data/court/queries.json"
OUTPUT="results/court/grid"

# Make sure codex (installed via npm to ~/.npm-global/bin) is reachable —
# non-interactive SSH doesn't source ~/.bashrc, so PATH must be set inline.
export PATH="$HOME/.npm-global/bin:$PATH"

# Azure key — extract from YAML key file (see docs/codex_setup.md).
if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  KEYFILE="${AZURE_KEY_FILE:-$HOME/api_keys/azure_cloudbank/gpt-54_1.txt}"
  if [[ -f "$KEYFILE" ]]; then
    export AZURE_OPENAI_API_KEY="$(awk -F': ' '/^api_key:/{print $2; exit}' "$KEYFILE")"
    echo "[sweep] AZURE_OPENAI_API_KEY loaded (${#AZURE_OPENAI_API_KEY} chars) from $KEYFILE"
  else
    echo "[sweep] ERROR: AZURE_OPENAI_API_KEY not set and $KEYFILE not found" >&2
    exit 1
  fi
fi

command -v codex >/dev/null || { echo "[sweep] ERROR: codex not on PATH" >&2; exit 1; }
echo "[sweep] codex: $(command -v codex)  ($(codex --version 2>/dev/null || echo unknown))"

mkdir -p logs/court_grid

run_combo() {
  local s="$1" g="$2" r="$3" a="$4"
  local tag="${s}_${g}_${r}_${a}"
  local log="logs/court_grid/${tag}.log"
  echo "=========================================================================="
  echo "[sweep] combo: sampling=$s  rule_gen=$g  refine=$r  apply=$a"
  echo "[sweep] log:   $log"
  echo "=========================================================================="
  python3 src/pipeline.py \
    --sampling-strategy "$s" \
    --rule-gen-strategy "$g" \
    --refine-strategy   "$r" \
    --apply-strategy    "$a" \
    --queries-file      "$QUERIES" \
    --dataset           "$DATASET" \
    --cluster           "$CLUSTER" \
    --output-dir        "$OUTPUT" \
    --skip-existing \
    2>&1 | tee "$log"
}

START=$(date +%s)
for s in random fps; do
  for g in llm_coarse_gpt54 agent_codex_gpt54; do
    for r in agentic_codex_gpt54 p_mini p_hybrid; do
      run_combo "$s" "$g" "$r" default
    done
  done
done
END=$(date +%s)

echo "=========================================================================="
echo "[sweep] DONE — wall time: $(( (END - START) / 60 )) minutes"
echo "[sweep] per-combo logs in logs/court_grid/"
echo "[sweep] aggregate results under $OUTPUT/apply/<sampling>/<rule_gen>/<refine>/default/"
