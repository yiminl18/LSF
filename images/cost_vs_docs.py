"""Generate cost vs. number of docs figure for baselines vs LSF."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Model prices ($/million tokens, input only) ───────────────────────────────
PRICES = {
    "opus47":    5.00,   # claude-opus-4-7
    "sonnet":    3.00,   # claude-sonnet-4-5
    "gpt54":     2.50,   # gpt-5.4
    "gpt54mini": 0.75,   # gpt-5.4-mini
}

# ── Data from screenshots ─────────────────────────────────────────────────────
# Baseline cost ratios  (input_tokens / avg_doc_tokens, per doc per question)
BASELINE_COST_RATIO = {
    "opus47":    0.9112,
    "sonnet":    1.8006,
    "gpt54":     1.3221,
    "gpt54mini": 1.2878,
}

# LSF cost ratios
LSF_RULE_GEN_RATIO    = 10.94   # one-time per question (gpt54)
LSF_RULE_REFINE_RATIO =  0.84   # one-time per question (gpt54) — agentic selection (Opus 4.7)
LSF_APPLY_RATIO       =  0.0090 # per (doc, question)   (gpt54 retrieval) — agentic+fallback

# ── Dataset constants ─────────────────────────────────────────────────────────
AVG_DOC_TOKENS = 88_432   # single-cluster, 10 sampled docs, tiktoken cl100k_base
N_QUESTIONS    = 1

# ── Helper ────────────────────────────────────────────────────────────────────
def price_per_token(model):
    return PRICES[model] / 1_000_000

def baseline_cost(n_docs, model):
    cost_per_pair = BASELINE_COST_RATIO[model] * AVG_DOC_TOKENS * price_per_token(model)
    return n_docs * N_QUESTIONS * cost_per_pair

def lsf_one_time_cost():
    ratio = LSF_RULE_GEN_RATIO + LSF_RULE_REFINE_RATIO
    return N_QUESTIONS * ratio * AVG_DOC_TOKENS * price_per_token("gpt54")

def lsf_cost(n_docs, apply_model):
    one_time = lsf_one_time_cost()
    cost_per_pair = LSF_APPLY_RATIO * AVG_DOC_TOKENS * price_per_token("gpt54")
    total = one_time + n_docs * N_QUESTIONS * cost_per_pair
    if apply_model == "gpt54mini":
        total = total / 6
    return total

# ── Accuracy labels (from screenshots) ───────────────────────────────────────
ACCURACY = {
    "opus47":       0.9466,
    "sonnet":       0.8788,
    "gpt54":        0.9102,
    "gpt54mini":    0.8800,
    "lsf_gpt54":    0.940,
    "lsf_gpt54mini":0.920,
}

# ── Plot ──────────────────────────────────────────────────────────────────────
n_docs = np.array([0, 50, 100, 150, 200])

fig, ax = plt.subplots(figsize=(11, 5.5))

# Baselines
baseline_styles = {
    "opus47":    dict(color="#E07B39", linestyle="-",  marker="o",
                     label=f"Baseline: Claude QA (opus47)  acc={ACCURACY['opus47']:.4f}"),
    "sonnet":    dict(color="#E8B800", linestyle="-",  marker="s",
                     label=f"Baseline: Claude QA (sonnet)  acc={ACCURACY['sonnet']:.4f}"),
    "gpt54":     dict(color="#5B8DD9", linestyle="-",  marker="^",
                     label=f"Baseline: Codex QA (gpt54)  acc={ACCURACY['gpt54']:.4f}"),
    "gpt54mini": dict(color="#5AAF5A", linestyle="-",  marker="D",
                     label=f"Baseline: Codex QA (gpt54mini)  acc={ACCURACY['gpt54mini']:.4f}"),
}
for model, style in baseline_styles.items():
    y = [baseline_cost(n, model) for n in n_docs]
    ax.plot(n_docs, y, linewidth=2, markersize=6, **style)
    ax.annotate(f"${y[-1]:,.0f}", xy=(200, y[-1]), xytext=(205, y[-1]),
                fontsize=8.5, va="center", color=style["color"], fontweight="bold",
                annotation_clip=False)

# LSF lines
lsf_gpt54_y = [lsf_cost(n, "gpt54")     for n in n_docs]
lsf_mini_y  = [lsf_cost(n, "gpt54mini") for n in n_docs]

ax.plot(n_docs, lsf_gpt54_y, color="#9B59B6", linestyle="--", marker="o", markersize=6,
        linewidth=2.5, label=f"LSF: rule apply w/ gpt54  acc={ACCURACY['lsf_gpt54']:.2f}")
ax.plot(n_docs, lsf_mini_y,  color="#E91E8C", linestyle="--", marker="s", markersize=6,
        linewidth=2.5, label=f"LSF: rule apply w/ gpt54mini  acc={ACCURACY['lsf_gpt54mini']:.2f}")
one_time = lsf_one_time_cost()
apply_200 = lsf_gpt54_y[-1] - one_time
ax.annotate(f"${one_time:,.1f}(one-time) + ${apply_200:,.1f}(apply) = ${lsf_gpt54_y[-1]:,.1f}",
            xy=(200, lsf_gpt54_y[-1]), xytext=(205, lsf_gpt54_y[-1]),
            fontsize=8.5, va="center", color="#9B59B6", fontweight="bold",
            annotation_clip=False)
one_time_mini = one_time / 6
apply_mini = apply_200 / 6
ax.annotate(f"${one_time_mini:,.1f}(one-time) + ${apply_mini:,.1f}(apply) = ${lsf_mini_y[-1]:,.1f}",
            xy=(200, lsf_mini_y[-1]), xytext=(205, lsf_mini_y[-1]),
            fontsize=8.5, va="center", color="#E91E8C", fontweight="bold",
            annotation_clip=False)

ax.set_xlabel("Number of documents", fontsize=12)
ax.set_ylabel("Total cost (USD)", fontsize=12)
ax.set_title("Cost vs. Number of Documents\n(1 question avg, input tokens only, single-cluster avg doc size)",
             fontsize=12)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlim(-5, 280)
plt.tight_layout()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/yiminglin/Documents/Codebase/LSF/images/cost_vs_docs.png", dpi=150)
print("Saved.")
