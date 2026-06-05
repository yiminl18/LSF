#!/usr/bin/env python3
"""Accuracy vs. cost-ratio scatter for all NOPV strategies (baselines + pipelines).

Data source: the normalized NOPV table in docs/final_result.md.
  - Accuracy  = (20*sAcc + 222*uAcc) / 242              (pipelines); correct-pair fraction (baselines)
  - Cost ratio= (20*(RL/20) + 222*unsampled_apply_cr)/242 (pipelines); per-pair token ratio (baselines)

Cost spans ~0.04 to ~34 (three orders of magnitude), so the y-axis is log-scaled.
Run:  python3 images/final_result/nopv/plot_nopv.py
Output: images/final_result/nopv/nopv_accuracy_vs_cost.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy, cost_ratio, group)  group in {baseline, llm_coarse, agent_codex}
DATA = [
    # baselines
    ("B1 Codex QA (per-pair) gpt54",      0.922, 33.89, "baseline"),
    ("B1 Codex QA (per-pair) gpt54mini",  0.880, 30.47, "baseline"),
    ("B2 Codex QA All gpt54",             0.867, 1.67,  "baseline"),
    ("B2 Codex QA All gpt54mini",         0.668, 2.42,  "baseline"),
    # pipelines — llm_coarse
    ("random/llm_coarse/p_mini",          0.935, 0.4743, "llm_coarse"),
    ("fps/llm_coarse/p_mini",             0.932, 0.4310, "llm_coarse"),
    ("fps/llm_coarse/agentic_codex",      0.931, 0.3327, "llm_coarse"),
    ("fps/llm_coarse/p_hybrid",           0.929, 0.2620, "llm_coarse"),
    ("random/llm_coarse/p_hybrid",        0.927, 0.4204, "llm_coarse"),
    ("random/llm_coarse/agentic_codex",   0.901, 0.3603, "llm_coarse"),
    # pipelines — agent_codex
    ("fps/agent_codex/p_hybrid",          0.853, 0.0422, "agent_codex"),
    ("fps/agent_codex/agentic_codex",     0.853, 0.0414, "agent_codex"),
    ("fps/agent_codex/p_mini",            0.852, 0.0422, "agent_codex"),
    ("random/agent_codex/p_mini",         0.846, 0.0410, "agent_codex"),
    ("random/agent_codex/p_hybrid",       0.846, 0.0410, "agent_codex"),
    ("random/agent_codex/agentic_codex",  0.843, 0.0397, "agent_codex"),
]

STYLE = {
    "baseline":    dict(color="#d62728", marker="X", s=130, label="Baseline (no rules)"),
    "llm_coarse":  dict(color="#1f77b4", marker="o", s=90,  label="Pipeline: llm_coarse"),
    "agent_codex": dict(color="#2ca02c", marker="s", s=90,  label="Pipeline: agent_codex"),
}

fig, ax = plt.subplots(figsize=(9, 6.5))

for group, st in STYLE.items():
    xs = [d[1] for d in DATA if d[3] == group]
    ys = [d[2] for d in DATA if d[3] == group]
    ax.scatter(xs, ys, edgecolors="black", linewidths=0.6, alpha=0.9, **st)

# label every point
for label, acc, cost, group in DATA:
    ax.annotate(label, (acc, cost), fontsize=6.5, xytext=(4, 3),
                textcoords="offset points", color="#333333")

ax.set_yscale("log")
ax.set_xlabel("Accuracy  (weighted over 20 sampled + 222 unsampled docs)", fontsize=11)
ax.set_ylabel("Cost ratio  (log scale; tokens / doc tokens, per doc)", fontsize=11)
ax.set_title("NOPV — Accuracy vs. Cost ratio (all strategies)", fontsize=13, fontweight="bold")
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.text(0.665, 1.08, "cost = 1 doc", fontsize=7, color="gray")
ax.legend(loc="center right", frameon=True, fontsize=9)

# annotate the desirable corner
ax.text(0.005, 0.02, "← lower cost, higher accuracy is better (bottom-right)",
        transform=ax.transAxes, fontsize=8, color="#555555")

fig.tight_layout()
out = Path(__file__).resolve().parent / "nopv_accuracy_vs_cost.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
