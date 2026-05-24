"""Rule selection for refine_dynamic_generality: one_shot → auto-tighten selected subset.

Reads one_shot rules and existing eval artefacts, runs the auto-tighten
selection algorithm (src/rule_refinement/select_rules_auto_tighten.py),
copies selected rule .py files into
rules/.../refine_dynamic_generality/<slug>_10_refdyn/, and writes selection
metadata to results/.../refine_dynamic_generality/rule_select/<slug>_10_refdyn.json.

Prerequisites (already produced by the one_shot pipeline):
    results/.../one_shot/eval_merge/<slug>_10_sampled.json   (Phase 1 D*)
    results/.../one_shot/eval_individual/<slug>_10_llm/      (Phase 3/4 tau)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from rule_refinement.cost_profile import load_or_compute_cost_profile
from rule_refinement.select_rules_auto_tighten import run_selection_auto_tighten

# ── Paths ──────────────────────────────────────────────────────────────────────
QUERIES_FILE         = "data/financebench/sample_queries.txt"
LABELS_FILE          = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
PROCESSING_DIR       = "data/financebench/processing"

ONE_SHOT_RULES_DIR   = "rules/financebench/lsf/single_cluster/llm/gpt54/one_shot"
ONE_SHOT_EVAL_MERGE  = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_merge"
ONE_SHOT_EVAL_IND    = "results/financebench/lsf/single_cluster/llm/gpt54/one_shot/eval_individual"

REFDYN_RULES_DIR     = "rules/financebench/lsf/single_cluster/llm/gpt54/refine_dynamic_generality"
REFDYN_RESULTS_BASE  = "results/financebench/lsf/single_cluster/llm/gpt54/refine_dynamic_generality"
COST_PROFILE_DIR     = f"{REFDYN_RESULTS_BASE}/cost_profile"
SELECTOR_RUN_DIR     = f"{REFDYN_RESULTS_BASE}/selector_run_auto"
RULE_SELECT_DIR      = f"{REFDYN_RESULTS_BASE}/rule_select"

TAU_FLOOR            = 0.0
EPSILON              = 1e-6
MAX_ITERS            = 10
MODEL_NAME           = "gpt54"


def make_slug(q: str) -> str:
    s = q.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


# ── Load shared data ───────────────────────────────────────────────────────────
questions = [l.strip() for l in open(QUERIES_FILE) if l.strip()]
labels: dict = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
doc_names = [k.replace(".pdf", "") for k in labels.keys()]

doc_map: dict[str, dict] = {}
for doc_name in doc_names:
    path = Path(PROCESSING_DIR) / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: missing {path}")

print(f"Questions: {len(questions)}  |  Docs: {len(doc_map)}")

for d in [REFDYN_RULES_DIR, COST_PROFILE_DIR, SELECTOR_RUN_DIR, RULE_SELECT_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

summary: list[dict] = []
total_wall_start = time.time()

# ── Per-question selection ─────────────────────────────────────────────────────
for question in questions:
    slug       = make_slug(question)
    rule_slug  = f"{slug}_10_llm"      # one_shot source folder
    refdyn_slug = f"{slug}_10_refdyn"  # destination folder

    out_file = Path(RULE_SELECT_DIR) / f"{refdyn_slug}.json"
    if out_file.exists():
        print(f"\nSKIP (exists): {out_file.name}")
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        summary.append(existing)
        continue

    rule_folder = Path(ONE_SHOT_RULES_DIR) / rule_slug
    if not rule_folder.is_dir():
        print(f"\nSKIP (no rule folder): {rule_folder}")
        continue

    eval_merge_path = Path(ONE_SHOT_EVAL_MERGE) / f"{slug}_10_sampled.json"
    if not eval_merge_path.exists():
        print(f"\nSKIP (no eval_merge): {eval_merge_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"Question : {question}")
    print(f"Source   : {rule_folder}")
    print(f"Dest     : {REFDYN_RULES_DIR}/{refdyn_slug}/")

    # Phase 0 — cost profile (cached per question; reuse from refine_generality if present)
    doc_paths = [
        Path(PROCESSING_DIR) / f"{d}_reconstructed.json"
        for d in doc_map
    ]
    cache_path = Path(COST_PROFILE_DIR) / f"{rule_slug}.json"
    cost_profile = load_or_compute_cost_profile(
        rules_dir=str(rule_folder),
        doc_paths=doc_paths,
        cache_path=cache_path,
    )
    print(f"  Cost profile: {len(cost_profile)} rules  (cached={cache_path.exists()})")

    eval_individual_dir = Path(ONE_SHOT_EVAL_IND) / rule_slug

    # Phases 2-4: run auto-tighten selection
    t_select_start = time.time()
    try:
        result = run_selection_auto_tighten(
            rules_dir=ONE_SHOT_RULES_DIR,
            eval_merge_path=eval_merge_path,
            eval_individual_dir=eval_individual_dir,
            documents=doc_map,
            question_slug=rule_slug,
            question=question,
            labels=labels,
            cost_profile=cost_profile,
            tau_floor=TAU_FLOOR,
            epsilon=EPSILON,
            max_iters=MAX_ITERS,
            model_name=MODEL_NAME,
            output_dir=SELECTOR_RUN_DIR,
        )
    except Exception as e:
        import traceback
        print(f"  FATAL ERROR in selection: {e}")
        traceback.print_exc()
        continue

    selection_latency = round(time.time() - t_select_start, 2)
    result["selection_latency_seconds"] = selection_latency

    selected_rules = result["selected_rules"]
    print(f"  Selected {len(selected_rules)} rules in {selection_latency:.1f}s")
    print(f"  Rules: {selected_rules}")
    print(f"  tau_floor={result['tau_floor']}  tau_best={result['tau_best']}")

    # Copy selected rule .py files to refine_dynamic_generality folder
    dest_folder = Path(REFDYN_RULES_DIR) / refdyn_slug
    dest_folder.mkdir(parents=True, exist_ok=True)

    copied = []
    for rule_name in selected_rules:
        src = rule_folder / f"{rule_name}.py"
        dst = dest_folder / f"{rule_name}.py"
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(rule_name)
        else:
            print(f"  WARNING: source rule not found: {src}")

    result["refdyn_slug"] = refdyn_slug
    result["refdyn_rules_dir"] = str(dest_folder)
    result["rules_copied"] = copied
    result["n_rules_total_pool"] = len(cost_profile)

    out_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"  Saved: {out_file.name}  "
        f"(covered={len(result['covered_docs'])}/{len(result['covered_docs']) + len(result['uncovered_docs'])}  "
        f"QA={result['llm_calls']['phase_2_incremental']}  "
        f"judge={result['llm_calls']['phase_3_coverage']}  "
        f"tighten={result['llm_calls']['phase_4_auto_tighten']})"
    )
    summary.append(result)

# ── Summary ────────────────────────────────────────────────────────────────────
total_wall = round(time.time() - total_wall_start, 1)
print(f"\n\nTotal wall time: {total_wall}s")
print(f"\n{'Question':<55}  {'Pool':>4}  {'Sel':>3}  {'TauBest':>7}  {'Cov':>5}  {'SelLat(s)':>9}  {'QACalls':>7}")
print("-" * 105)
for s in summary:
    n_pool  = s.get("n_rules_total_pool", "?")
    n_sel   = len(s.get("selected_rules", []))
    tau_b   = s.get("tau_best", 0.0)
    n_cov   = len(s.get("covered_docs", []))
    n_tot   = n_cov + len(s.get("uncovered_docs", []))
    lat     = s.get("selection_latency_seconds", 0.0)
    qa      = s.get("llm_calls", {}).get("phase_2_incremental", 0)
    q_short = s.get("question", "")[:53]
    print(
        f"  {q_short:<53}  {n_pool!s:>4}  {n_sel:>3}  "
        f"{tau_b:>7.3f}  {n_cov}/{n_tot}  {lat:>9.1f}  {qa:>7}"
    )

summary_path = Path(RULE_SELECT_DIR) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nRule files : {REFDYN_RULES_DIR}/")
print(f"Metadata   : {RULE_SELECT_DIR}/")
print(f"Summary    : {summary_path}")
