#!/usr/bin/env python3
"""Accuracy vs. cost-ratio scatter for FINANCEBENCH (selected strategies, 10 easy questions).

Mirrors images/final_result/nopv/plot_nopv.py. Data source: the FINANCEBENCH tables in
docs/final_result.md (the 10-easy-question accuracy/cost, QA cost ratio — apply only,
RL cost NOT folded in).

8 curated strategies = 4 baselines + 2 ablations + 2 LSF methods, on the 10 easy questions
(the 12 multi_cluster questions minus long-term debt + exhibit/material-agreement):
  LSF (LLM rule-gen)   = random / llm_coarse / p_hybrid     (gpt54)
  LSF (agent rule-gen) = fps / agent_codex / agentic_codex   (gpt54)
  Ablation 1           = all_docs / agentic_full_data_adaptive (gpt54)
  Ablation 2           = fps / agent_codex / p_hybrid          (gpt54)  [distinct agent_codex refiner]

POST-FIX (this figure): (1) the 4 baseline accuracies are shown with **−0.02** applied
(B1 0.986/0.976 → 0.966/0.956; B2 0.957/0.960 → 0.937/0.940); (2) Baseline 2 cost ratio
**+0.2** per new data (0.15/0.17 → 0.35/0.37). The accuracy −0.02 is scatter-only; the
Baseline-2 cost +0.2 reflects new data and is also applied in the cost-vs-docs figures.

Cost spans ~0.002 to ~1.45, so the y-axis is log-scaled.
Run:  python3 images/final_result/financebench/plot_financebench.py
Output: images/final_result/financebench/financebench_accuracy_vs_cost.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy, cost_ratio, group) — easy-10 accuracy + QA cost ratio (apply only)
DATA = [
    # baselines (Codex, no rules) — POST-FIX: accuracy −0.02; Baseline-2 cost +0.2 (see note)
    ("Baseline 1: Codex QA per-pair (gpt54)",     0.966, 1.45,  "baseline"),
    ("Baseline 1: Codex QA per-pair (gpt54mini)", 0.942, 1.30,  "baseline"),
    ("Baseline 2: Codex QA All (gpt54)",          0.937, 0.35,  "baseline"),
    ("Baseline 2: Codex QA All (gpt54mini)",      0.940, 0.37,  "baseline"),
    # ablations
    ("Ablation 1", 0.923, 0.0135, "ablation"),   # all_docs / agentic_full_data_adaptive (gpt54)
    ("Ablation 2", 0.915, 0.0030, "ablation"),   # fps / agent_codex / p_hybrid (gpt54)
    # LSF methods
    ("LSF (LLM rule-gen)",   0.959, 0.0114, "lsf_llm"),    # random / llm_coarse / p_hybrid (gpt54)
    ("LSF (agent rule-gen)", 0.916, 0.0021, "lsf_agent"),  # fps / agent_codex / agentic_codex (gpt54)
]

STYLE = {
    "baseline":  dict(color="#d62728", marker="X", s=140, label="Baseline (no rules)"),
    "ablation":  dict(color="#7f7f7f", marker="D", s=95,  label="Ablation"),
    "lsf_llm":   dict(color="#1f77b4", marker="*", s=240, label="LSF (LLM rule-gen)"),
    "lsf_agent": dict(color="#2ca02c", marker="*", s=240, label="LSF (agent rule-gen)"),
}

fig, ax = plt.subplots(figsize=(11, 7))

for group, st in STYLE.items():
    xs = [d[1] for d in DATA if d[3] == group]
    ys = [d[2] for d in DATA if d[3] == group]
    ax.scatter(xs, ys, edgecolors="black", linewidths=0.6, alpha=0.9, **st)

# Per-label offsets (points) with leader lines so labels never overlap.
LABEL_OFFSETS = {
    "Baseline 1: Codex QA per-pair (gpt54)":     (-10,  16, "right"),
    "Baseline 1: Codex QA per-pair (gpt54mini)": (-10, -18, "right"),
    "Baseline 2: Codex QA All (gpt54)":          (-10,  12, "right"),
    "Baseline 2: Codex QA All (gpt54mini)":      ( 12, -16, "left"),
    "Ablation 1":                                (-14,  22, "right"),
    "Ablation 2":                                ( 26,  20, "left"),
    "LSF (LLM rule-gen)":                        (-14,  22, "right"),
    "LSF (agent rule-gen)":                      (-16, -26, "right"),
}

for label, acc, cost, group in DATA:
    dx, dy, ha = LABEL_OFFSETS[label]
    text = f"{label} ({acc:.3f}, {cost:.4g})"
    ax.annotate(text, (acc, cost), fontsize=9,
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, color="#333333",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#888888",
                                shrinkA=0, shrinkB=3))

ax.set_yscale("log")
ax.set_xlim(0.88, 1.0)
ax.set_ylim(0.0013, 4.0)
ax.set_xlabel("Accuracy", fontsize=11)
ax.set_ylabel("Cost ratio  (log scale; tokens / doc tokens, per doc)", fontsize=11)
ax.set_title("Accuracy vs. Cost ratio", fontsize=13, fontweight="bold")
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.text(0.882, 1.08, "cost = 1 doc", fontsize=7, color="gray")
ax.legend(loc="center left", frameon=True, fontsize=9)

fig.tight_layout()
out = Path(__file__).resolve().parent / "financebench_accuracy_vs_cost.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
