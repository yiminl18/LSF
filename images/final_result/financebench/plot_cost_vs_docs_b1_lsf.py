#!/usr/bin/env python3
"""FINANCEBENCH — cost vs. #docs for the two COMPARABLE-ACCURACY strategies only.

Mirrors images/final_result/nopv/plot_cost_vs_docs_b1_lsf.py. Same cost model
(cost = one-time RL + n*per-doc apply, gpt54 input price), restricted to the two
strategies with comparable accuracy on financebench:

  - Baseline 1 — Agentic Codex QA (per-pair), gpt54        acc 0.966  (−0.02 post-fix)
  - LSF (LLM rule-gen) — random/llm_coarse/p_hybrid (gpt54) acc 0.959

Emits BOTH a log-y and a linear-y version.

⚠ RL note (same as plot_cost_vs_docs.py): the financebench grid did not separately log
rule-gen tokens, so LSF-LLM's one-time RL is ESTIMATED as one pass over the ~18-doc sample
(18 × JSON-prompt doc size). Baseline 1 has no rule learning.

Run:    python3 images/final_result/financebench/plot_cost_vs_docs_b1_lsf.py
Output: images/final_result/financebench/financebench_cost_vs_docs_b1_lsf.png         (log y)
        images/final_result/financebench/financebench_cost_vs_docs_b1_lsf_linear.png  (linear y)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

PRICE_GPT54   = 2.50 / 1_000_000
DOC_TOK_PLAIN = 65_633
DOC_TOK_JSON  = 29_444
N_QUESTIONS   = 1
N_DOCS_MAX    = 86

# Two comparable-accuracy strategies only.
STRATS = [
    ("Baseline 1: Codex QA per-pair (gpt54)", 0.966, 0.0,                  1.45  * DOC_TOK_PLAIN,
        dict(color="#5B8DD9", ls="-", marker="^")),
    ("LSF (LLM rule-gen): random/llm_coarse/p_hybrid", 0.959, 18 * DOC_TOK_JSON, 0.0114 * DOC_TOK_PLAIN,
        dict(color="#9B59B6", ls="-", marker="o")),
]

def cost_usd(n, ot, pd):
    return N_QUESTIONS * ot * PRICE_GPT54 + n * N_QUESTIONS * pd * PRICE_GPT54

N_DOCS_EST = 1000        # extrapolated estimate point (linear cost model), to scale
x_line  = np.linspace(0, N_DOCS_EST, 600)
x_marks = np.array([0, 20, 40, 60, N_DOCS_MAX])

def make(logy: bool, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    handles = []
    for label, acc, ot, pd, st in STRATS:
        meas = x_line <= N_DOCS_MAX
        ax.plot(x_line[meas], cost_usd(x_line[meas], ot, pd), linewidth=2.4, color=st["color"], ls=st["ls"])
        ax.plot(x_line[~meas], cost_usd(x_line[~meas], ot, pd), linewidth=2.0, color=st["color"], ls=":", alpha=0.8)
        ym = cost_usd(x_marks, ot, pd)
        vis = ym > 0 if logy else np.ones_like(ym, dtype=bool)
        ax.scatter(x_marks[vis], ym[vis], s=52, marker=st["marker"], color=st["color"], zorder=3)
        y_est = cost_usd(N_DOCS_EST, ot, pd)
        ax.scatter([N_DOCS_EST], [y_est], s=72, marker=st["marker"],
                   facecolors="white", edgecolors=st["color"], linewidths=1.8, zorder=3)
        handles.append(Line2D([0], [0], color=st["color"], ls=st["ls"], marker=st["marker"],
                              markersize=8, linewidth=2.4, label=f"{label}  (acc={acc:.3f})"))
        ax.annotate(f"${ym[-1]:,.2f}", xy=(N_DOCS_MAX, ym[-1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=8.5, va="center",
                    color=st["color"], fontweight="bold", annotation_clip=False)
        ax.annotate(f"${y_est:,.2f} (est.)", xy=(N_DOCS_EST, y_est), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center",
                    color=st["color"], fontweight="bold", annotation_clip=False)
    if logy:
        ax.set_yscale("log")
        ylab = "Total cost (USD, log scale) — RL + apply"; scale_txt = "log y"
    else:
        ax.set_ylim(bottom=0)
        ylab = "Total cost (USD) — RL + apply"; scale_txt = "linear y"
    ax.set_xlabel("Number of documents", fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title("FINANCEBENCH — Cost vs. Number of Documents (comparable-accuracy strategies)\n"
                 f"(1 question avg, gpt54 input price; cost = one-time RL + per-doc apply; {scale_txt})",
                 fontsize=12)
    ax.set_xticks([0, N_DOCS_MAX, 200, 400, 600, 800, N_DOCS_EST])
    ax.set_xlim(-15, 1120)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:,.2f}"))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(handles=handles, fontsize=9.5, loc="upper left" if not logy else "lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

here = Path(__file__).resolve().parent
make(logy=True,  out=here / "financebench_cost_vs_docs_b1_lsf.png")
make(logy=False, out=here / "financebench_cost_vs_docs_b1_lsf_linear.png")
