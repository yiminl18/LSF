#!/usr/bin/env python3
"""FINANCEBENCH — total cost (USD) vs. number of documents — LINEAR y-axis variant.

Identical to plot_cost_vs_docs.py (same 6 gpt54 strategies, same cost model
cost(n) = one-time RL + n * per-doc apply, all on the 10 easy questions) EXCEPT the
y-axis is linear. Same RL caveat (grid RL estimated; see plot_cost_vs_docs.py header).

Run:    python3 images/final_result/financebench/plot_cost_vs_docs_linear.py
Output: images/final_result/financebench/financebench_cost_vs_docs_linear.png
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

LLM_COARSE_RL_TOK  = 18 * DOC_TOK_JSON
AGENT_CODEX_RL_TOK = 1.0 * DOC_TOK_JSON
ABLATION1_RL_TOK   = (36_531_055 + 328_785) / 12

STRATS = [
    ("Baseline 1: Codex QA per-pair (gpt54)", 0.986, 0.0,                1.45  * DOC_TOK_PLAIN,
        dict(color="#5B8DD9", ls="-",  marker="^")),
    ("Baseline 2: Codex QA All (gpt54)",      0.957, 0.0,                0.15  * DOC_TOK_PLAIN,
        dict(color="#5AAF5A", ls="-",  marker="D")),
    ("Ablation 1: agentic_full_data_adaptive", 0.967, ABLATION1_RL_TOK,  0.0135 * DOC_TOK_PLAIN,
        dict(color="#7f7f7f", ls="-.", marker="o")),
    ("Ablation 2: fps/agent_codex/p_hybrid",  0.915, AGENT_CODEX_RL_TOK, 0.0030 * DOC_TOK_PLAIN,
        dict(color="#555555", ls="-.", marker="s")),
    ("LSF (LLM rule-gen): random/llm_coarse/p_hybrid", 0.953, LLM_COARSE_RL_TOK, 0.0114 * DOC_TOK_PLAIN,
        dict(color="#9B59B6", ls="--", marker="o")),
    ("LSF (agent rule-gen): fps/agent_codex/agentic_codex", 0.916, AGENT_CODEX_RL_TOK, 0.0021 * DOC_TOK_PLAIN,
        dict(color="#1f9e89", ls="--", marker="*")),
]

def cost_usd(n, ot, pd):
    return N_QUESTIONS * ot * PRICE_GPT54 + n * N_QUESTIONS * pd * PRICE_GPT54

x_line  = np.linspace(0, N_DOCS_MAX, 400)
x_marks = np.array([0, 20, 40, 60, N_DOCS_MAX])
fig, ax = plt.subplots(figsize=(11, 6))

legend_handles = []
for label, acc, ot, pd, st in STRATS:
    ax.plot(x_line, cost_usd(x_line, ot, pd), linewidth=2.2, color=st["color"], ls=st["ls"])
    ym = cost_usd(x_marks, ot, pd)
    ax.scatter(x_marks, ym, s=46, marker=st["marker"], color=st["color"], zorder=3)
    legend_handles.append(Line2D([0], [0], color=st["color"], ls=st["ls"], marker=st["marker"],
                                 markersize=8, linewidth=2.2, label=f"{label}  (acc={acc:.3f})"))
    ax.annotate(f"${ym[-1]:,.2f}", xy=(N_DOCS_MAX, ym[-1]), xytext=(6, 0),
                textcoords="offset points", fontsize=8, va="center",
                color=st["color"], fontweight="bold", annotation_clip=False)

# linear y-axis (the only change vs plot_cost_vs_docs.py)
ax.set_xlabel("Number of documents", fontsize=12)
ax.set_ylabel("Total cost (USD) — RL + apply", fontsize=12)
ax.set_title("FINANCEBENCH — Cost vs. Number of Documents (10 easy questions, linear y)\n"
             "(1 question avg, gpt54 input price; cost = one-time rule-learning + per-doc apply)",
             fontsize=12)
ax.set_xticks([0, 20, 40, 60, N_DOCS_MAX])
ax.set_xlim(-2, 100)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.2f}"))
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left")
fig.text(0.5, 0.012,
         "⚠ Grid RL not separately logged: llm_coarse RL estimated as ~1 pass over the 18-doc sample; "
         "agent_codex RL ≈1 doc (per nopv); agentic_full_data RL measured but full-price (no finance cache split).",
         ha="center", fontsize=7, color="#777777")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path(__file__).resolve().parent / "financebench_cost_vs_docs_linear.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
