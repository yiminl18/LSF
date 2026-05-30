#!/usr/bin/env bash
# Run the 4 recommended court-winning pipelines on the NOPV dataset, in parallel.
#
# The 4 pipelines (sampling / rule_gen / refine / apply=default):
#   fps    / agent_codex_gpt54 / agentic_codex_gpt54   (court best overall)
#   random / agent_codex_gpt54 / agentic_codex_gpt54   (best without fps)
#   fps    / llm_coarse_gpt54  / p_hybrid              (best non-Codex-gen)
#   random / llm_coarse_gpt54  / agentic_codex_gpt54   (cheapest llm_coarse)
#
# Each has a DISTINCT (sampling, rule_gen) pair, so they write to disjoint paths
# and are safe to run concurrently. Every invocation uses --skip-existing.
#
# NOPV specifics: 242 docs, 12 questions, doc JSONs live in data/nopv/json/
# (processing/ is empty), split derived at run time from data/nopv/all_labels.json.

set -uo pipefail
cd "$(dirname "$0")/.."

DATASET="nopv"
CLUSTER="all_docs"
QUERIES="data/nopv/queries.json"
PROC="data/nopv/json"
OUTPUT="results/nopv/grid"
NQ=12   # nopv question count, for the completeness check

# codex (npm global) + Azure key — non-interactive SSH doesn't source ~/.bashrc.
export PATH="$HOME/.npm-global/bin:$PATH"
if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  KEYFILE="${AZURE_KEY_FILE:-$HOME/api_keys/azure_cloudbank/gpt-54_1.txt}"
  if [[ -f "$KEYFILE" ]]; then
    export AZURE_OPENAI_API_KEY="$(awk -F': ' '/^api_key:/{print $2; exit}' "$KEYFILE")"
    echo "[nopv] AZURE_OPENAI_API_KEY loaded (${#AZURE_OPENAI_API_KEY} chars)"
  else
    echo "[nopv] ERROR: AZURE_OPENAI_API_KEY not set and $KEYFILE not found" >&2; exit 1
  fi
fi
command -v codex >/dev/null || { echo "[nopv] ERROR: codex not on PATH" >&2; exit 1; }
echo "[nopv] codex: $(command -v codex)"

mkdir -p logs/nopv_top4

COMBOS=(
  "fps    agent_codex_gpt54 agentic_codex_gpt54"
  "random agent_codex_gpt54 agentic_codex_gpt54"
  "fps    llm_coarse_gpt54  p_hybrid"
  "random llm_coarse_gpt54  agentic_codex_gpt54"
)

echo "=========================================================================="
echo "[nopv] launching ${#COMBOS[@]} pipelines in parallel"
echo "=========================================================================="
START=$(date +%s)
for combo in "${COMBOS[@]}"; do
  read -r s g r <<< "$combo"
  log="logs/nopv_top4/${s}_${g}_${r}.log"
  echo "[nopv] start: $s / $g / $r  -> $log"
  python3 src/pipeline.py \
    --sampling-strategy "$s" \
    --rule-gen-strategy "$g" \
    --refine-strategy   "$r" \
    --apply-strategy    default \
    --queries-file      "$QUERIES" \
    --dataset           "$DATASET" \
    --cluster           "$CLUSTER" \
    --processing-dir    "$PROC" \
    --output-dir        "$OUTPUT" \
    --skip-existing \
    > "$log" 2>&1 &
done
wait
END=$(date +%s)

echo "=========================================================================="
echo "[nopv] all pipelines finished — wall time: $(( (END - START) / 60 )) min"
echo "[nopv] --- apply completeness check (X/$NQ unsampled) ---"
WARN=0
for combo in "${COMBOS[@]}"; do
  read -r s g r <<< "$combo"
  n=$(find "$OUTPUT/apply/$s/$g/$r/default" -name "*_unsampled.json" 2>/dev/null | wc -l | tr -d ' ')
  sm="$OUTPUT/apply/$s/$g/$r/default/pipeline_summary.json"
  if [[ "$n" -ge "$NQ" && -s "$sm" ]]; then
    echo "  ✓ $s/$g/$r : $n/$NQ"
  else
    echo "  ⚠️ $s/$g/$r : $n/$NQ (incomplete — see logs/nopv_top4/${s}_${g}_${r}.log)"
    WARN=1
  fi
done
echo "=========================================================================="
[[ "$WARN" -eq 1 ]] && { echo "⚠️ [nopv] one or more pipelines incomplete"; exit 2; }
echo "✓ [nopv] all 4 pipelines complete"
exit 0
