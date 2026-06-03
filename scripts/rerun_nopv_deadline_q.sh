#!/usr/bin/env bash
# One-off: re-run ONLY the "longest deadline" question for the nopv pipeline
# fps/agent_codex_gpt54/agentic_codex_gpt54, after the empty-selection fallback fix.
#
# The agent had selected 0 rules from a 5-rule pool for this question (misjudged
# the labels) -> empty refined dir -> apply dropped the question (11/12). We clear
# ONLY that question's stale refined artifacts so refine re-runs (now with the
# full-pool fallback); every other question and pipeline is left untouched
# (--skip-existing skips the 11 already-complete questions).

set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.npm-global/bin:$PATH"
export AZURE_OPENAI_API_KEY="$(awk -F': ' '/^api_key:/{print $2; exit}' "$HOME/api_keys/azure_cloudbank/gpt-54_1.txt")"

Q="what_is_the_longest_deadline_in_days_measured_from_the_final"
B="results/nopv/grid/refined/fps/agent_codex_gpt54/agentic_codex_gpt54"
rm -rf "$B/$Q" "$B/$Q.json" "$B/${Q}_refine.json"
echo "[rerun] cleared stale refined artifacts for: $Q"

python3 src/pipeline.py \
  --sampling-strategy fps \
  --rule-gen-strategy agent_codex_gpt54 \
  --refine-strategy   agentic_codex_gpt54 \
  --apply-strategy    default \
  --queries-file      data/nopv/queries.json \
  --dataset           nopv \
  --cluster           all_docs \
  --processing-dir    data/nopv/json \
  --output-dir        results/nopv/grid \
  --stop-after        apply \
  --skip-existing

echo "[rerun] done. apply files for this pipeline:"
ls results/nopv/grid/apply/fps/agent_codex_gpt54/agentic_codex_gpt54/default/*_unsampled.json 2>/dev/null | wc -l
