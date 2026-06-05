#!/usr/bin/env python3
"""Accuracy vs. cost-ratio scatter for OFFICEQA (baselines + the 6 agent_codex pipelines).

Data source: the OFFICEQA section in docs/final_result.md.
  NOTE: only agent_codex pipelines exist (llm_coarse overflows context on these huge docs).
      Accuracy   = (20*sAcc + 180*uAcc) / 200
      Cost ratio = (20*(RL/20) + 180*unsampled_apply_cr) / 200   (pipelines)
                 = per-pair / amortized input/doc token ratio     (baselines)

Cost spans ~0.0034 to ~164 (≈5 orders of magnitude), so the y-axis is log-scaled.
Run:  python3 images/final_result/officeqa/plot_officeqa.py
Output: images/final_result/officeqa/officeqa_accuracy_vs_cost.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy, cost_ratio, group)
DATA = [
    # baselines
    ("B1 Codex QA (per-pair) gpt54",     0.830, 160.04, "baseline"),
    ("B1 Codex QA (per-pair) gpt54mini", 0.796, 164.41, "baseline"),
    ("B2 Codex QA All gpt54",            0.779, 20.86,  "baseline"),
    ("B2 Codex QA All gpt54mini",        0.556, 9.87,   "baseline"),
    # pipelines (agent_codex only)
    ("random/agent_codex/p_mini",        0.585, 0.0074, "agent_codex"),
    ("random/agent_codex/p_hybrid",      0.584, 0.0074, "agent_codex"),
    ("fps/agent_codex/p_hybrid",         0.572, 0.0036, "agent_codex"),
    ("fps/agent_codex/p_mini",           0.572, 0.0036, "agent_codex"),
    ("fps/agent_codex/agentic_codex",    0.569, 0.0034, "agent_codex"),
    ("random/agent_codex/agentic_codex", 0.555, 0.0063, "agent_codex"),
]

STYLE = {
    "baseline":    dict(color="#d62728", marker="X", s=130, label="Baseline (no rules)"),
    "agent_codex": dict(color="#2ca02c", marker="s", s=95,  label="Pipeline: agent_codex"),
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
ax.set_xlabel("Accuracy  (weighted over 20 sampled + 180 unsampled docs)", fontsize=11)
ax.set_ylabel("Cost ratio  (log scale; tokens / doc tokens, per doc)", fontsize=11)
ax.set_title("OFFICEQA — Accuracy vs. Cost ratio (all strategies)", fontsize=13, fontweight="bold")
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.text(0.56, 1.25, "cost = 1 doc", fontsize=7, color="gray")
ax.legend(loc="center right", frameon=True, fontsize=9)
ax.text(0.005, 0.02, "← lower cost, higher accuracy is better (bottom-right)",
        transform=ax.transAxes, fontsize=8, color="#555555")

fig.tight_layout()
out = Path(__file__).resolve().parent / "officeqa_accuracy_vs_cost.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
