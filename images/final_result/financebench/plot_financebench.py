#!/usr/bin/env python3
"""Accuracy vs. cost-ratio scatter for FINANCEBENCH (baselines + legacy pipelines).

Data source: the FINANCEBENCH section in docs/final_result.md.
  NOTE: financebench has NO 12-combo grid. These are 5 pre-existing legacy pipelines
  (single_cluster / multi_clusters). RL (rule-learning) tokens were not recorded for them,
  so the cost ratio shown is the COMBINED APPLY cost only:
      Accuracy   = (ns*sAcc + nu*uAcc) / (ns+nu)
      Cost ratio = combined apply retrieved/doc over both splits
  Baselines: per-pair / amortized input/doc token ratio.

Cost spans ~0.014 to ~1.74, so the y-axis is log-scaled.
Run:  python3 images/final_result/financebench/plot_financebench.py
Output: images/final_result/financebench/financebench_accuracy_vs_cost.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy, cost_ratio, group)
DATA = [
    # baselines (Codex)
    ("B1 Codex QA (per-pair) gpt54",     0.931, 1.45, "baseline"),
    ("B1 Codex QA (per-pair) gpt54mini", 0.878, 1.30, "baseline"),
    ("B2 Codex QA All gpt54",            0.861, 0.15, "baseline"),
    ("B2 Codex QA All gpt54mini",        0.820, 0.17, "baseline"),
    # reference baselines (Claude)
    ("Claude QA opus47",                 0.947, 0.91, "baseline_claude"),
    ("Claude QA sonnet",                 0.895, 1.74, "baseline_claude"),
    # legacy pipelines
    ("multi/llm_coarse/raw gpt54",       0.846, 0.0829, "pipeline"),
    ("single/agent/raw opus47",          0.794, 0.1217, "pipeline"),
    ("single/llm_coarse/raw gpt54",      0.772, 0.1076, "pipeline"),
    ("single/agent/refined gpt54",       0.758, 0.0364, "pipeline"),
    ("multi/llm_coarse/raw opus47",      0.716, 0.0137, "pipeline"),
]

STYLE = {
    "baseline":        dict(color="#d62728", marker="X", s=130, label="Baseline (Codex, no rules)"),
    "baseline_claude": dict(color="#9467bd", marker="P", s=120, label="Baseline (Claude, ref)"),
    "pipeline":        dict(color="#2ca02c", marker="s", s=95,  label="LSF pipeline (legacy)"),
}

fig, ax = plt.subplots(figsize=(9, 6.5))
for group, st in STYLE.items():
    xs = [d[1] for d in DATA if d[3] == group]
    ys = [d[2] for d in DATA if d[3] == group]
    ax.scatter(xs, ys, edgecolors="black", linewidths=0.6, alpha=0.9, **st)

for label, acc, cost, group in DATA:
    ax.annotate(label, (acc, cost), fontsize=6.5, xytext=(4, 3),
                textcoords="offset points", color="#333333")

ax.set_yscale("log")
ax.set_xlabel("Accuracy  (weighted over sampled + unsampled docs)", fontsize=11)
ax.set_ylabel("Cost ratio  (log scale; apply tokens / doc tokens, per doc)", fontsize=11)
ax.set_title("FINANCEBENCH — Accuracy vs. Cost ratio (legacy pipelines + baselines)",
             fontsize=12.5, fontweight="bold")
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.text(0.72, 1.08, "cost = 1 doc", fontsize=7, color="gray")
ax.legend(loc="center left", frameon=True, fontsize=9)
ax.text(0.005, 0.02, "← lower cost, higher accuracy is better (bottom-right);  RL cost not included (unrecorded)",
        transform=ax.transAxes, fontsize=7.5, color="#555555")

fig.tight_layout()
out = Path(__file__).resolve().parent / "financebench_accuracy_vs_cost.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
