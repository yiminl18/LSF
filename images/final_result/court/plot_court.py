#!/usr/bin/env python3
"""Accuracy vs. cost-ratio scatter for all COURT strategies (baselines + pipelines).

Data source: the normalized COURT table in docs/final_result.md.
  - Accuracy   = (20*sAcc + 274*uAcc) / 294               (pipelines); correct-pair fraction (baselines)
  - Cost ratio = (20*(RL/20) + 274*unsampled_apply_cr)/294 (pipelines); per-pair token ratio (baselines)

Cost spans ~0.005 to ~33 (≈4 orders of magnitude), so the y-axis is log-scaled.
Run:  python3 images/final_result/court/plot_court.py
Output: images/final_result/court/court_accuracy_vs_cost.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy, cost_ratio, group)  group in {baseline, llm_coarse, agent_codex}
DATA = [
    # baselines
    ("B1 Codex QA (per-pair) gpt54",      0.909, 33.32, "baseline"),
    ("B1 Codex QA (per-pair) gpt54mini",  0.906, 30.80, "baseline"),
    ("B2 Codex QA All gpt54",             0.880, 1.33,  "baseline"),
    ("B2 Codex QA All gpt54mini",         0.867, 1.61,  "baseline"),
    # pipelines — agent_codex
    ("fps/agent_codex/agentic_codex",     0.925, 0.0051, "agent_codex"),
    ("fps/agent_codex/p_mini",            0.925, 0.0055, "agent_codex"),
    ("fps/agent_codex/p_hybrid",          0.925, 0.0055, "agent_codex"),
    ("random/agent_codex/p_hybrid",       0.911, 0.0059, "agent_codex"),
    ("random/agent_codex/p_mini",         0.910, 0.0059, "agent_codex"),
    ("random/agent_codex/agentic_codex",  0.909, 0.0052, "agent_codex"),
    # pipelines — llm_coarse
    ("fps/llm_coarse/p_mini",             0.874, 0.1162, "llm_coarse"),
    ("fps/llm_coarse/p_hybrid",           0.873, 0.0919, "llm_coarse"),
    ("fps/llm_coarse/agentic_codex",      0.867, 0.0953, "llm_coarse"),
    ("random/llm_coarse/p_mini",          0.846, 0.0964, "llm_coarse"),
    ("random/llm_coarse/p_hybrid",        0.846, 0.0917, "llm_coarse"),
    ("random/llm_coarse/agentic_codex",   0.845, 0.0840, "llm_coarse"),
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

for label, acc, cost, group in DATA:
    ax.annotate(label, (acc, cost), fontsize=6.5, xytext=(4, 3),
                textcoords="offset points", color="#333333")

ax.set_yscale("log")
ax.set_xlabel("Accuracy  (weighted over 20 sampled + 274 unsampled docs)", fontsize=11)
ax.set_ylabel("Cost ratio  (log scale; tokens / doc tokens, per doc)", fontsize=11)
ax.set_title("COURT — Accuracy vs. Cost ratio (all strategies)", fontsize=13, fontweight="bold")
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.text(0.845, 1.1, "cost = 1 doc", fontsize=7, color="gray")
ax.legend(loc="center right", frameon=True, fontsize=9)
ax.text(0.005, 0.02, "← lower cost, higher accuracy is better (bottom-right)",
        transform=ax.transAxes, fontsize=8, color="#555555")

fig.tight_layout()
out = Path(__file__).resolve().parent / "court_accuracy_vs_cost.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
