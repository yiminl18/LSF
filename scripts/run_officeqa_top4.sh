#!/usr/bin/env bash
# Run the 4 court-recommended pipelines (same as run_nopv_top4.sh) on officeqa,
# full pipelines (sampling -> rule_gen -> [precompute] -> refine -> apply=default).
#
#   fps    / agent_codex_gpt54 / agentic_codex_gpt54
#   random / agent_codex_gpt54 / agentic_codex_gpt54
#   fps    / llm_coarse_gpt54  / p_hybrid
#   random / llm_coarse_gpt54  / agentic_codex_gpt54
#
# Distinct (sampling, rule_gen) pairs -> disjoint paths -> safe in parallel.
# --skip-existing reuses the rule pools + fps/random sampling already produced by
# the officeqa rule-gen run; this adds precompute (for p_hybrid) + refine + apply.
#
# officeqa specifics: docs read from data/officeqa/normalized_json (converted
# texts-schema); 200 docs, 16 questions, split derived from all_labels.json.

set -uo pipefail
cd "$(dirname "$0")/.."

DATASET="officeqa"
CLUSTER="all_docs"
QUERIES="data/officeqa/queries.json"
PROC="data/officeqa/normalized_json"
OUTPUT="results/officeqa/grid"
NQ=16

export PATH="$HOME/.npm-global/bin:$PATH"
if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  KEYFILE="${AZURE_KEY_FILE:-$HOME/api_keys/azure_cloudbank/gpt-54_1.txt}"
  if [[ -f "$KEYFILE" ]]; then
    export AZURE_OPENAI_API_KEY="$(awk -F': ' '/^api_key:/{print $2; exit}' "$KEYFILE")"
    echo "[oqa-top4] AZURE_OPENAI_API_KEY loaded (${#AZURE_OPENAI_API_KEY} chars)"
  else
    echo "[oqa-top4] ERROR: AZURE_OPENAI_API_KEY not set and $KEYFILE not found" >&2; exit 1
  fi
fi
command -v codex >/dev/null || { echo "[oqa-top4] ERROR: codex not on PATH" >&2; exit 1; }
echo "[oqa-top4] codex: $(command -v codex)"

mkdir -p logs/officeqa_top4

COMBOS=(
  "fps    agent_codex_gpt54 agentic_codex_gpt54"
  "random agent_codex_gpt54 agentic_codex_gpt54"
  "fps    llm_coarse_gpt54  p_hybrid"
  "random llm_coarse_gpt54  agentic_codex_gpt54"
)

echo "=========================================================================="
echo "[oqa-top4] launching ${#COMBOS[@]} full pipelines in parallel"
echo "=========================================================================="
START=$(date +%s)
for combo in "${COMBOS[@]}"; do
  read -r s g r <<< "$combo"
  log="logs/officeqa_top4/${s}_${g}_${r}.log"
  echo "[oqa-top4] start: $s / $g / $r  -> $log"
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
echo "[oqa-top4] done — wall time: $(( (END - START) / 60 )) min"
echo "[oqa-top4] --- apply completeness (X/$NQ unsampled) ---"
WARN=0
for combo in "${COMBOS[@]}"; do
  read -r s g r <<< "$combo"
  n=$(find "$OUTPUT/apply/$s/$g/$r/default" -name "*_unsampled.json" 2>/dev/null | wc -l | tr -d ' ')
  sm="$OUTPUT/apply/$s/$g/$r/default/pipeline_summary.json"
  if [[ "$n" -ge "$NQ" && -s "$sm" ]]; then
    echo "  ✓ $s/$g/$r : $n/$NQ"
  else
    echo "  ⚠️ $s/$g/$r : $n/$NQ (incomplete — see logs/officeqa_top4/${s}_${g}_${r}.log)"
    WARN=1
  fi
done
echo "=========================================================================="
[[ "$WARN" -eq 1 ]] && { echo "⚠️ [oqa-top4] one or more pipelines incomplete"; exit 2; }
echo "✓ [oqa-top4] all 4 pipelines complete"
exit 0
