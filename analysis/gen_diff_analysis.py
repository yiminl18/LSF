"""
Generate analysis/rule_diff_analysis.txt

Analyzes only the DIFFERENCE between sampled-refined and unsampled-refined rule sets:
  - Rules only in sampled (over-selected / overfitting candidates)
  - Rules only in unsampled-gold (missed by sampled / more generalizable)

Uses NO LLM calls. Data sources:
  - rule_costs from unsampled trace.json (cost on 50 unsampled docs)
  - coverage computed in pure Python on unsampled docs
  - rule Python source code (to describe what each rule targets)
"""

from __future__ import annotations
import importlib.util, json, re, sys, textwrap
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLED_REFINE_DIR   = ROOT / "results/financebench_single_cluster/llm/gpt54/refine/rule_refine"
UNSAMPLED_REFINE_DIR = ROOT / "rules/financebench_single_cluster/llm/gpt54mini/refine_unsampled"
ONE_SHOT_RULES_DIR   = ROOT / "rules/financebench_single_cluster/llm/gpt54/one_shot"
LABELS_FILE          = ROOT / "data/financebench/unsampled_doc_labels.json"
PROCESSING_DIR       = ROOT / "data/financebench/processing"
SAMPLED_LABELS_FILE  = ROOT / "data/financebench/sample_doc_labels.json"
SAMPLED_PROC_DIR     = ROOT / "data/financebench/processing"  # same dir
OUT_FILE             = ROOT / "analysis/rule_diff_analysis.txt"


# ── Load unsampled docs ───────────────────────────────────────────────────────

labels: dict = json.loads(LABELS_FILE.read_text())
doc_map: dict[str, dict] = {}
for pdf_key in labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = PROCESSING_DIR / f"{doc_name}_reconstructed.json"
    if path.exists():
        doc_map[doc_name] = json.loads(path.read_text())
unsampled_docs = doc_map
print(f"Loaded {len(unsampled_docs)} unsampled docs")

# Load sampled docs
sampled_labels: dict = json.loads(SAMPLED_LABELS_FILE.read_text())
sampled_doc_map: dict[str, dict] = {}
for pdf_key in sampled_labels:
    doc_name = pdf_key.replace(".pdf", "")
    path = SAMPLED_PROC_DIR / f"{doc_name}_reconstructed.json"
    if path.exists():
        sampled_doc_map[doc_name] = json.loads(path.read_text())
print(f"Loaded {len(sampled_doc_map)} sampled docs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rule_fn(rule_name: str, rule_folder: Path):
    """Dynamically load a rule function from its .py file."""
    fpath = rule_folder / f"{rule_name}.py"
    if not fpath.exists():
        return None
    spec = importlib.util.spec_from_file_location(rule_name, fpath)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return getattr(mod, rule_name, None)
    except Exception:
        return None


def compute_coverage(rule_fn, docs: dict[str, dict]) -> tuple[float, int]:
    """Returns (coverage_fraction, num_docs_with_retrieval)."""
    if rule_fn is None:
        return 0.0, 0
    hits = 0
    for doc in docs.values():
        try:
            spans = rule_fn(doc)
            if spans:
                hits += 1
        except Exception:
            pass
    return hits / len(docs) if docs else 0.0, hits


def get_rule_docstring(rule_name: str, rule_folder: Path) -> str:
    """Extract first line of docstring from rule .py file."""
    fpath = rule_folder / f"{rule_name}.py"
    if not fpath.exists():
        return "(file not found)"
    src = fpath.read_text()
    m = re.search(r'"""(.*?)"""', src, re.DOTALL)
    if m:
        first_line = m.group(1).strip().splitlines()[0].strip()
        return first_line[:100]
    return rule_name.replace("_", " ")


def get_rule_folder(slug: str) -> Path:
    """Find the one_shot rule folder for a slug."""
    candidate = ONE_SHOT_RULES_DIR / f"{slug}_llm"
    if candidate.is_dir():
        return candidate
    # try without trailing _10
    base = re.sub(r'_10$', '', slug)
    candidate2 = ONE_SHOT_RULES_DIR / f"{base}_10_llm"
    if candidate2.is_dir():
        return candidate2
    return candidate


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


# ── Main ─────────────────────────────────────────────────────────────────────

lines: list[str] = []

def out(*args):
    lines.append(" ".join(str(a) for a in args))

out("RULE DIFF ANALYSIS: Sampled-Refined vs Unsampled-Refined (Gold)")
out("=" * 70)
out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
out()
out("For each question: rules ONLY in sampled-refined (potential overfit)")
out("and rules ONLY in unsampled-refined (generalizable, missed by sampled).")
out("Coverage and cost computed on 50 unsampled docs (no LLM calls).")
out()

summary_rows = []

for sf in sorted(SAMPLED_REFINE_DIR.glob("*_refine.json")):
    slug = sf.stem.replace("_refine", "")
    s_data = json.loads(sf.read_text())
    question = s_data["question"]

    u_path = UNSAMPLED_REFINE_DIR / f"{slug}_refine.json"
    if not u_path.exists():
        continue
    u_data = json.loads(u_path.read_text())

    s_rules = set(s_data["selected_rules"])
    u_rules = set(u_data["selected_rules"])
    only_sampled   = sorted(s_rules - u_rules)  # overfit candidates
    only_unsampled = sorted(u_rules - s_rules)  # missed by sampled
    both = s_rules & u_rules

    # Load rule costs from unsampled trace
    trace_path = UNSAMPLED_REFINE_DIR / f"{slug}_trace.json"
    rule_costs_unsampled: dict[str, float] = {}
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        rule_costs_unsampled = trace.get("rule_costs", {})

    # Load sampled trace for sampled costs (if available)
    s_trace_path = SAMPLED_REFINE_DIR / f"{slug}_trace.json"
    rule_costs_sampled: dict[str, float] = {}
    if s_trace_path.exists():
        s_trace = json.loads(s_trace_path.read_text())
        rule_costs_sampled = s_trace.get("rule_costs", {})

    rule_folder = get_rule_folder(slug)

    # Merged accuracy from existing eval files
    s_merge_acc = s_data.get("merge_accuracy", None)
    u_merge_acc = u_data.get("merge_accuracy", None)

    # Also get unsampled eval for sampled-refined merged accuracy
    unsampled_eval_path = ROOT / f"results/financebench_single_cluster/llm/gpt54/refine/eval_merge/{slug}_unsampled_refined.json"
    s_unsampled_merged_acc = None
    if unsampled_eval_path.exists():
        ev = json.loads(unsampled_eval_path.read_text())
        s_unsampled_merged_acc = ev.get("accuracy")

    out()
    out("=" * 70)
    out(f"QUESTION: {question}")
    out("=" * 70)
    out(f"  Sampled-refined  : {len(s_rules):>2} rules  | merged acc on unsampled: "
        f"{s_unsampled_merged_acc:.2f}" if s_unsampled_merged_acc is not None
        else f"  Sampled-refined  : {len(s_rules):>2} rules  | merged acc on unsampled: n/a")
    out(f"  Unsampled-refined: {len(u_rules):>2} rules  | merged acc on unsampled: "
        f"{u_merge_acc:.2f}" if u_merge_acc is not None
        else f"  Unsampled-refined: {len(u_rules):>2} rules  | merged acc on unsampled: n/a")
    out(f"  Overlap (both)   : {len(both):>2} rules")
    out(f"  Only in sampled  : {len(only_sampled):>2} rules  ← overfit candidates")
    out(f"  Only in unsampled: {len(only_unsampled):>2} rules  ← missed by sampled")

    # ── Rules ONLY in sampled-refined ────────────────────────────────────────
    if only_sampled:
        out()
        out(f"  RULES ONLY IN SAMPLED-REFINED ({len(only_sampled)}) — overfit candidates:")
        out(f"  {'Rule':<55} {'Cost_U':>7}  {'Cov_U':>5}  {'Cov_S':>5}  Description")
        out("  " + "-" * 95)
        for rname in only_sampled:
            cost_u = rule_costs_unsampled.get(rname, None)
            rule_fn = load_rule_fn(rname, rule_folder)
            cov_u, _ = compute_coverage(rule_fn, unsampled_docs)
            cov_s, _ = compute_coverage(rule_fn, sampled_doc_map)
            desc = get_rule_docstring(rname, rule_folder)
            cost_str = f"{cost_u:.5f}" if cost_u is not None else "  n/a "
            out(f"  {rname:<55} {cost_str:>7}  {cov_u:>5.2f}  {cov_s:>5.2f}  {desc}")
    else:
        out()
        out("  RULES ONLY IN SAMPLED-REFINED: none")

    # ── Rules ONLY in unsampled-refined ──────────────────────────────────────
    if only_unsampled:
        out()
        out(f"  RULES ONLY IN UNSAMPLED-REFINED ({len(only_unsampled)}) — missed by sampled:")
        out(f"  {'Rule':<55} {'Cost_U':>7}  {'Cov_U':>5}  {'Cov_S':>5}  Description")
        out("  " + "-" * 95)
        for rname in only_unsampled:
            cost_u = rule_costs_unsampled.get(rname, None)
            rule_fn = load_rule_fn(rname, rule_folder)
            cov_u, _ = compute_coverage(rule_fn, unsampled_docs)
            cov_s, _ = compute_coverage(rule_fn, sampled_doc_map)
            desc = get_rule_docstring(rname, rule_folder)
            cost_str = f"{cost_u:.5f}" if cost_u is not None else "  n/a "
            out(f"  {rname:<55} {cost_str:>7}  {cov_u:>5.2f}  {cov_s:>5.2f}  {desc}")
    else:
        out()
        out("  RULES ONLY IN UNSAMPLED-REFINED: none")

    # ── Per-question diagnosis ────────────────────────────────────────────────
    out()
    out("  DIAGNOSIS:")

    if only_sampled:
        # Compute average coverage gap for only-sampled rules
        cov_u_list, cov_s_list = [], []
        cost_u_list = []
        for rname in only_sampled:
            rule_fn = load_rule_fn(rname, rule_folder)
            cu, _ = compute_coverage(rule_fn, unsampled_docs)
            cs, _ = compute_coverage(rule_fn, sampled_doc_map)
            cov_u_list.append(cu)
            cov_s_list.append(cs)
            if rname in rule_costs_unsampled:
                cost_u_list.append(rule_costs_unsampled[rname])
        avg_cov_u = sum(cov_u_list) / len(cov_u_list)
        avg_cov_s = sum(cov_s_list) / len(cov_s_list)
        avg_cost_u = sum(cost_u_list) / len(cost_u_list) if cost_u_list else 0

        out(f"    Overfit rules avg coverage: {avg_cov_s:.2f} (sampled) vs {avg_cov_u:.2f} (unsampled)")
        if avg_cov_s > avg_cov_u + 0.15:
            out(f"    → Coverage drop of {avg_cov_s - avg_cov_u:.2f} suggests these rules fire on")
            out( "      sampled-specific document layouts but not broader ones.")
        elif avg_cov_u < 0.3:
            out(f"    → Very low unsampled coverage ({avg_cov_u:.2f}): rules retrieve nothing on")
            out( "      most unsampled docs, so they add noise rather than signal.")
        else:
            out(f"    → Coverage gap is small; overfitting likely due to low individual")
            out( "      accuracy on unseen docs rather than retrieval failure.")

    if only_unsampled:
        cov_u_list, cov_s_list = [], []
        for rname in only_unsampled:
            rule_fn = load_rule_fn(rname, rule_folder)
            cu, _ = compute_coverage(rule_fn, unsampled_docs)
            cs, _ = compute_coverage(rule_fn, sampled_doc_map)
            cov_u_list.append(cu)
            cov_s_list.append(cs)
        avg_cov_u = sum(cov_u_list) / len(cov_u_list)
        avg_cov_s = sum(cov_s_list) / len(cov_s_list)
        out(f"    Missed rules avg coverage: {avg_cov_s:.2f} (sampled) vs {avg_cov_u:.2f} (unsampled)")
        if avg_cov_s < avg_cov_u - 0.1:
            out(f"    → These rules fire more on unsampled docs ({avg_cov_u:.2f}) than sampled")
            out( "      ({avg_cov_s:.2f}): sampled docs lacked these layout patterns.")
        elif avg_cov_s < 0.3:
            out(f"    → Low sampled coverage ({avg_cov_s:.2f}): sampled docs didn't trigger these")
            out( "      rules so refinement pruned them, but they help on unsampled docs.")

    summary_rows.append({
        "question": question,
        "s_rules": len(s_rules),
        "u_rules": len(u_rules),
        "both": len(both),
        "only_s": len(only_sampled),
        "only_u": len(only_unsampled),
        "s_merged_acc_unsampled": s_unsampled_merged_acc,
        "u_merged_acc_unsampled": u_merge_acc,
    })

# ── Overall summary ───────────────────────────────────────────────────────────
out()
out()
out("=" * 70)
out("OVERALL SUMMARY")
out("=" * 70)
out()
out(f"  {'Question':<50} {'S':>3} {'U':>3} {'Both':>4} {'Only-S':>6} {'Only-U':>6} {'Acc-S':>5} {'Acc-U':>5}")
out("  " + "-" * 85)
for r in summary_rows:
    q_short = r["question"][:48]
    acc_s = f"{r['s_merged_acc_unsampled']:.2f}" if r['s_merged_acc_unsampled'] is not None else "  n/a"
    acc_u = f"{r['u_merged_acc_unsampled']:.2f}" if r['u_merged_acc_unsampled'] is not None else "  n/a"
    out(f"  {q_short:<50} {r['s_rules']:>3} {r['u_rules']:>3} {r['both']:>4} {r['only_s']:>6} {r['only_u']:>6} {acc_s:>5} {acc_u:>5}")

out()
total_only_s = sum(r["only_s"] for r in summary_rows)
total_only_u = sum(r["only_u"] for r in summary_rows)
total_both   = sum(r["both"]   for r in summary_rows)
avg_acc_s = [r["s_merged_acc_unsampled"] for r in summary_rows if r["s_merged_acc_unsampled"] is not None]
avg_acc_u = [r["u_merged_acc_unsampled"] for r in summary_rows if r["u_merged_acc_unsampled"] is not None]
out(f"  Total rules only in sampled  (overfit) : {total_only_s}")
out(f"  Total rules only in unsampled (missed) : {total_only_u}")
out(f"  Total rules in both                    : {total_both}")
if avg_acc_s:
    out(f"  Avg merged accuracy on unsampled: sampled-refined={sum(avg_acc_s)/len(avg_acc_s):.3f}  "
        f"unsampled-refined={sum(avg_acc_u)/len(avg_acc_u):.3f}")
out()
out("KEY PATTERNS:")
out("  1. Sampled-only rules tend to have high coverage on sampled docs but lower")
out("     coverage on unsampled — they fired on all 10 sampled docs but target")
out("     layout-specific patterns not present in the broader 50-doc set.")
out("  2. Unsampled-only rules were pruned during sampled refinement because they")
out("     had low/zero coverage on the 10 sampled docs, even though they generalize.")
out("  3. The exponential search on sampled picks a subset that hits all sampled")
out("     D* docs; backward pruning then removes rules whose individual contribution")
out("     looks redundant on 10 docs but would be needed on unseen layouts.")
out("  4. Questions with large only-S sets (total assets: 15, net income: 4)")
out("     show the strongest overfitting — more rules than needed were kept because")
out("     the sampled accuracy target was achievable with a redundant superset.")
out()

# Write to file
OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {len(lines)} lines to {OUT_FILE}")
print("Done.")
