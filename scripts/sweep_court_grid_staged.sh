#!/usr/bin/env bash
# Staged sweep of the 12-combo court grid — same combos as sweep_court_grid.sh,
# but driven STAGE-BY-STAGE across the whole grid, with per-branch gating.
#
# Why staged: every stage is idempotent (--skip-existing reuses on-disk output),
# so this produces the SAME results for the SAME cost as the combo-first sweep.
# The win is debuggability: each stage runs as one step, and after every combo's
# stage runs we CHECK its output. If the check passes, that branch continues to
# the next stage. If it FAILS, we PRUNE that branch — its downstream stages are
# skipped — and record a warning. Other branches that passed keep going.
#
# Example: if rule_gen for (random, agent_codex_gpt54) produces an empty pool,
# we skip Stage 3+ for that lineage and warn, but (random, llm_coarse_gpt54)
# continues into refinement normally.
#
# Stage order (pipeline.py --stop-after):
#   1. sampling    — once per sampling strategy            key: <s>
#   2. rule_gen    — once per (sampling, rule_gen)          key: <s>/<g>
#   3. precompute  — once per (sampling, rule_gen)          feeds Pareto refiners only
#   4. refine      — once per (sampling, rule_gen, refine)  key: <s>/<g>/<r>
#   5. apply       — once per full combo
#
# Grid (matches sweep_court_grid.sh):
#   sampling : random, fps
#   rule_gen : llm_coarse_gpt54, agent_codex_gpt54
#   refine   : agentic_codex_gpt54, p_mini, p_hybrid
#   apply    : default
#
# Exit status: 0 if every branch completed clean; 2 if any branch was pruned.

set -uo pipefail   # NOT -e: a failed check must prune + continue, not abort
cd "$(dirname "$0")/.."

DATASET="court"
CLUSTER="all_docs"
QUERIES="data/court/queries.json"
OUTPUT="results/court/grid"
RULES_ROOT="rules/$DATASET/grid"

SAMPLINGS=(random fps)
RULEGENS=(llm_coarse_gpt54 agent_codex_gpt54)
REFINES=(agentic_codex_gpt54 p_mini p_hybrid)
# Subset of REFINES that consume the Phase C precompute caches. A precompute
# failure prunes only these refine branches (agentic_codex does not need it).
PRECOMPUTE_REFINES=(p_mini p_hybrid)
APPLY="default"

# codex (npm global) + Azure key — non-interactive SSH doesn't source ~/.bashrc.
export PATH="$HOME/.npm-global/bin:$PATH"
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

mkdir -p logs/court_grid_staged

# ── Branch-pruning state ────────────────────────────────────────────────────
DEAD=$'\n'          # newline-delimited set of pruned branch keys
PRUNED_REPORT=""    # human-readable list for the final summary

mark_dead() {       # mark_dead <key> <reason>
  case "$DEAD" in *$'\n'"$1"$'\n'*) ;; *) DEAD+="$1"$'\n';; esac
  PRUNED_REPORT+="  ⚠️  $1 — $2"$'\n'
  echo "⚠️  [PRUNE] $1 — $2 (skipping downstream stages for this branch)" >&2
}
is_dead() { case "$DEAD" in *$'\n'"$1"$'\n'*) return 0;; *) return 1;; esac; }

# pruned <s> [g] [r] — true if this branch or any ancestor was pruned.
pruned() {
  local s="$1" g="${2:-}" r="${3:-}"
  is_dead "$s" && return 0
  [[ -n "$g" ]] && is_dead "$s/$g" && return 0
  [[ -n "$r" ]] && is_dead "$s/$g/$r" && return 0
  return 1
}

# ── Output checks (return 0 = ok, 1 = empty/missing) ────────────────────────
have_files() {      # have_files <dir> <glob> [min]
  local dir="$1" pat="$2" min="${3:-1}" n=0
  [[ -d "$dir" ]] && n=$(find "$dir" -type f -name "$pat" 2>/dev/null | wc -l | tr -d ' ')
  (( n >= min ))
}

# ── Stage runner ────────────────────────────────────────────────────────────
run() {             # run <stop_after> <sampling> <rule_gen> <refine>
  local stop="$1" s="$2" g="$3" r="$4"
  local log="logs/court_grid_staged/${stop}__${s}_${g}_${r}.log"
  echo "--------------------------------------------------------------------------"
  echo "[stage:$stop] sampling=$s rule_gen=$g refine=$r  -> $log"
  python3 src/pipeline.py \
    --sampling-strategy "$s" \
    --rule-gen-strategy "$g" \
    --refine-strategy   "$r" \
    --apply-strategy    "$APPLY" \
    --queries-file      "$QUERIES" \
    --dataset           "$DATASET" \
    --cluster           "$CLUSTER" \
    --output-dir        "$OUTPUT" \
    --stop-after        "$stop" \
    --skip-existing \
    2>&1 | tee "$log"
}

START=$(date +%s)

# ── STEP 1 — Sampling ───────────────────────────────────────────────────────
echo "=========================================================================="
echo "[sweep] STEP 1 — Sampling"
echo "=========================================================================="
for s in "${SAMPLINGS[@]}"; do
  run sampling "$s" "${RULEGENS[0]}" "${REFINES[1]}"
  if have_files "$OUTPUT/sampling/$s" "*.json" 2; then
    echo "✓  [CHECK:OK] sampling/$s"
  else
    mark_dead "$s" "sampling produced < 2 label files"
  fi
done

# ── STEP 2 — Rule generation ────────────────────────────────────────────────
echo "=========================================================================="
echo "[sweep] STEP 2 — Rule generation"
echo "=========================================================================="
for s in "${SAMPLINGS[@]}"; do
  pruned "$s" && { echo "[skip] $s/* rule_gen — sampling pruned"; continue; }
  for g in "${RULEGENS[@]}"; do
    run rule_gen "$s" "$g" "${REFINES[1]}"
    if have_files "$RULES_ROOT/$s/$g" "*.py"; then
      echo "✓  [CHECK:OK] rule_gen $s/$g"
    else
      mark_dead "$s/$g" "rule_gen produced an empty rule pool"
    fi
  done
done

# ── STEP 3 — Precompute (feeds Pareto refiners only) ────────────────────────
echo "=========================================================================="
echo "[sweep] STEP 3 — Precompute"
echo "=========================================================================="
for s in "${SAMPLINGS[@]}"; do
  for g in "${RULEGENS[@]}"; do
    pruned "$s" "$g" && { echo "[skip] $s/$g precompute — branch pruned"; continue; }
    run precompute "$s" "$g" p_mini
    if have_files "$OUTPUT/cache/$s/$g/cost_profile" "*.json" \
       && have_files "$OUTPUT/cache/$s/$g/eval_merge_base" "*.json" \
       && have_files "$OUTPUT/cache/$s/$g/eval_individual_gpt54mini" "*_eval.json"; then
      echo "✓  [CHECK:OK] precompute $s/$g"
    else
      # Prune only the precompute-dependent refine branches; agentic_codex survives.
      for r in "${PRECOMPUTE_REFINES[@]}"; do
        mark_dead "$s/$g/$r" "precompute caches incomplete (needed by $r)"
      done
    fi
  done
done

# ── STEP 4 — Refinement ─────────────────────────────────────────────────────
echo "=========================================================================="
echo "[sweep] STEP 4 — Refinement"
echo "=========================================================================="
for s in "${SAMPLINGS[@]}"; do
  for g in "${RULEGENS[@]}"; do
    for r in "${REFINES[@]}"; do
      pruned "$s" "$g" "$r" && { echo "[skip] $s/$g/$r refine — branch pruned"; continue; }
      run refine "$s" "$g" "$r"
      if have_files "$OUTPUT/refined/$s/$g/$r" "*.py"; then
        echo "✓  [CHECK:OK] refine $s/$g/$r"
      else
        mark_dead "$s/$g/$r" "refine produced an empty selected subset"
      fi
    done
  done
done

# ── STEP 5 — Apply + Evaluate ───────────────────────────────────────────────
echo "=========================================================================="
echo "[sweep] STEP 5 — Apply + Evaluate"
echo "=========================================================================="
for s in "${SAMPLINGS[@]}"; do
  for g in "${RULEGENS[@]}"; do
    for r in "${REFINES[@]}"; do
      pruned "$s" "$g" "$r" && { echo "[skip] $s/$g/$r apply — branch pruned"; continue; }
      run apply "$s" "$g" "$r"
      if [[ -s "$OUTPUT/apply/$s/$g/$r/$APPLY/pipeline_summary.json" ]]; then
        echo "✓  [CHECK:OK] apply $s/$g/$r"
      else
        mark_dead "$s/$g/$r" "apply produced no pipeline_summary.json"
      fi
    done
  done
done

END=$(date +%s)
echo "=========================================================================="
echo "[sweep] DONE — wall time: $(( (END - START) / 60 )) minutes"
echo "[sweep] per-step logs in logs/court_grid_staged/"
echo "[sweep] summaries under $OUTPUT/apply/<sampling>/<rule_gen>/<refine>/$APPLY/pipeline_summary.json"
if [[ -n "$PRUNED_REPORT" ]]; then
  echo "=========================================================================="
  echo "⚠️  [sweep] COMPLETED WITH PRUNED BRANCHES:"
  printf '%s' "$PRUNED_REPORT"
  echo "    (downstream stages for the above were skipped — investigate their logs)"
  echo "=========================================================================="
  exit 2
fi
echo "✓  [sweep] All branches completed clean."
exit 0
