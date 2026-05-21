"""Plot multi-cluster one-shot gpt54 results: accuracy and cost comparison.

Three strategies compared:
  1. Full pool — sampled  (18 docs, eval_merge/perf_summary.json)
  2. Full pool — unsampled (68 docs, eval_merge/perf_summary.json)
  3. Agentic selection + fallback — unsampled (68 docs, eval_agentic_fallback/summary.json)

Cost metric:
  Full pool: avg_cost_ratio = retrieved_tokens / doc_tokens  (lower = cheaper)
  Agentic fallback: cost_per_doc_usd computed from total_tokens and Azure pricing
    gpt54      = $2.50/M input,  $10.00/M output
    gpt54mini  = $0.15/M input,  $0.60/M output
  Full pool cost_per_doc_usd estimated using avg_cost_ratio * avg_doc_tokens * gpt54 price.
  avg_doc_tokens estimated at 47,819 (sampled from data/financebench/processing/).
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

FULL_POOL_FILE  = _ROOT / "results/financebench_multi_clusters/llm/gpt54/one_shot/eval_merge/perf_summary.json"
FALLBACK_FILE   = _ROOT / "results/financebench_multi_clusters/llm/gpt54/one_shot/eval_agentic_fallback/summary.json"
OUT_PNG         = Path(__file__).parent / "multi_cluster_results.png"

# Azure pricing ($ per token)
P_GPT54_IN    = 2.50  / 1_000_000
P_GPT54_OUT   = 10.00 / 1_000_000
P_MINI_IN     = 0.15  / 1_000_000
P_MINI_OUT    = 0.60  / 1_000_000

AVG_DOC_TOKENS = 47_819   # estimated from financebench processing JSONs


def short_label(q: str) -> str:
    q = q.strip()
    mapping = {
        "What stock exchange": "Stock exchange",
        "What is the reporting period": "Reporting period",
        "What is the exact name of the company": "Company name (exact)",
        "What is the company's principal executive offices address": "Office address (city/state)",
        "What document type": "Doc type (form)",
        "What is/are the trading symbol": "Trading symbols",
        "What is the registrant's exact name": "Registrant name",
        "What is the address of principal executive offices": "Office address (ZIP)",
        "What is the registrant's telephone": "Phone number",
        "What is the state (or other jurisdiction)": "State / IRS EIN",
        "What is long-term debt": "Long-term debt",
        "List one material agreement": "Exhibit listing",
    }
    for prefix, label in mapping.items():
        if q.startswith(prefix):
            return label
    return q[:30]


def load_full_pool() -> dict[str, dict]:
    rows = json.loads(FULL_POOL_FILE.read_text())
    out = {}
    for r in rows:
        q = r["question"]
        sampled_cost_usd   = r["sampled_avg_cost_ratio"]   * AVG_DOC_TOKENS * P_GPT54_IN
        unsampled_cost_usd = r["unsampled_avg_cost_ratio"] * AVG_DOC_TOKENS * P_GPT54_IN
        out[q] = {
            "sampled_acc":        r["sampled_accuracy"],
            "unsampled_acc":      r["unsampled_accuracy"],
            "sampled_cost_ratio": r["sampled_avg_cost_ratio"],
            "unsampled_cost_ratio": r["unsampled_avg_cost_ratio"],
            "sampled_cost_usd":   sampled_cost_usd,
            "unsampled_cost_usd": unsampled_cost_usd,
        }
    return out


def load_fallback() -> dict[str, dict]:
    rows = json.loads(FALLBACK_FILE.read_text())
    out = {}
    for r in rows:
        q    = r["question"]
        toks = r["total_tokens"]
        n    = r.get("num_documents", 68)
        cost_total = (
            toks["gpt54mini_in"]  * P_MINI_IN  +
            toks["gpt54mini_out"] * P_MINI_OUT +
            toks["gpt54_in"]      * P_GPT54_IN  +
            toks["gpt54_out"]     * P_GPT54_OUT
        )
        cost_per_doc = cost_total / n
        out[q] = {
            "acc":          r["accuracy"],
            "fallback_rate": r["fallback_rate"],
            "cost_per_doc": cost_per_doc,
            "gpt54_in_per_doc": toks["gpt54_in"] / n,
        }
    return out


def main():
    full  = load_full_pool()
    fallb = load_fallback()

    # Align on questions present in both, sorted by fallback accuracy desc
    common_q = [q for q in fallb if q in full]
    common_q.sort(key=lambda q: fallb[q]["acc"], reverse=True)

    labels = [short_label(q) for q in common_q]
    n = len(labels)
    x = np.arange(n)
    w = 0.26

    # ── Accuracy ─────────────────────────────────────────────────────────────
    acc_full_s  = [full[q]["sampled_acc"]   for q in common_q]
    acc_full_u  = [full[q]["unsampled_acc"] for q in common_q]
    acc_fallb   = [fallb[q]["acc"]          for q in common_q]

    # ── Cost ($/doc) ─────────────────────────────────────────────────────────
    cost_full_s = [full[q]["sampled_cost_usd"]   for q in common_q]
    cost_full_u = [full[q]["unsampled_cost_usd"] for q in common_q]
    cost_fallb  = [fallb[q]["cost_per_doc"]      for q in common_q]

    # ── Cost ratio (retrieved fraction) for full pool ─────────────────────
    cr_full_s = [full[q]["sampled_cost_ratio"]   for q in common_q]
    cr_full_u = [full[q]["unsampled_cost_ratio"] for q in common_q]
    # Agentic+fallback cost ratio: gpt54_in_per_doc / avg_doc_tokens
    # (same unit as full pool; gpt54mini_in is ~10x cheaper so shown separately)
    cr_fallb  = [fallb[q]["gpt54_in_per_doc"] / AVG_DOC_TOKENS for q in common_q]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle("Multi-Cluster One-Shot GPT-4o: Full Pool vs Agentic+Fallback", fontsize=14, fontweight="bold")

    C1, C2, C3 = "#4C72B0", "#DD8452", "#55A868"

    # ── Plot 1: Accuracy ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - w, acc_full_s, width=w, label="Full pool — sampled (18)",   color=C1, alpha=0.85)
    ax.bar(x,     acc_full_u, width=w, label="Full pool — unsampled (68)", color=C2, alpha=0.85)
    ax.bar(x + w, acc_fallb,  width=w, label="Agentic+fallback — unsampled (68)", color=C3, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.axhline(np.mean(acc_full_s), color=C1, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(acc_full_u), color=C2, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(acc_fallb),  color=C3, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"Accuracy  |  mean: full-pool-s={np.mean(acc_full_s):.3f}  full-pool-u={np.mean(acc_full_u):.3f}  agentic+fb={np.mean(acc_fallb):.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 2: Cost ratio (retrieved fraction) ───────────────────────────────
    ax = axes[1]
    ax.bar(x - w, cr_full_s, width=w, label="Full pool — sampled",            color=C1, alpha=0.85)
    ax.bar(x,     cr_full_u, width=w, label="Full pool — unsampled",           color=C2, alpha=0.85)
    ax.bar(x + w, cr_fallb,  width=w, label="Agentic+fallback — unsampled\n(gpt54_in_per_doc / doc_tokens)", color=C3, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Cost ratio\n(retrieved_tokens / doc_tokens)")
    ax.axhline(np.mean(cr_full_s), color=C1, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(cr_full_u), color=C2, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(np.mean(cr_fallb),  color=C3, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(
        f"Cost Ratio  |  mean: full-pool-s={np.mean(cr_full_s):.4f}  "
        f"full-pool-u={np.mean(cr_full_u):.4f}  agentic+fb={np.mean(cr_fallb):.4f}"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Plot 3: $/doc — fallback vs estimated full pool unsampled ─────────────
    ax = axes[2]
    ax.bar(x - w/2, [c * 1000 for c in cost_full_u], width=w, label="Full pool — unsampled (est.)", color=C2, alpha=0.85)
    ax.bar(x + w/2, [c * 1000 for c in cost_fallb],  width=w, label="Agentic+fallback — unsampled", color=C3, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Est. cost (m$/doc)\ngpt54 only for full pool; gpt54+mini for fallback")
    ax.set_title(
        f"Cost per Doc (m$)  |  mean full-pool={np.mean(cost_full_u)*1000:.4f}  "
        f"agentic+fb={np.mean(cost_fallb)*1000:.4f}"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Fallback rate annotation on plot 3 ────────────────────────────────────
    fb_rates = [fallb[q]["fallback_rate"] for q in common_q]
    for i, (xi, rate) in enumerate(zip(x, fb_rates)):
        ax.text(xi + w/2, cost_fallb[i]*1000 + 0.0002, f"{rate:.0%}", ha="center", fontsize=6, color="darkgreen")

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'Question':<42} {'sAcc':>6} {'uAcc':>6} {'fbAcc':>6} {'fbRate':>7} {'sCR':>7} {'uCR':>7} {'fbCR':>7}")
    print("-" * 100)
    for q, fb_cr in zip(common_q, cr_fallb):
        lbl = short_label(q)
        print(f"{lbl:<42} {full[q]['sampled_acc']:>6.3f} {full[q]['unsampled_acc']:>6.3f} "
              f"{fallb[q]['acc']:>6.3f} {fallb[q]['fallback_rate']:>7.1%} "
              f"{full[q]['sampled_cost_ratio']:>7.4f} {full[q]['unsampled_cost_ratio']:>7.4f} {fb_cr:>7.4f}")
    print("-" * 100)
    print(f"{'MEAN':<42} {np.mean(acc_full_s):>6.3f} {np.mean(acc_full_u):>6.3f} "
          f"{np.mean(acc_fallb):>6.3f} {np.mean(fb_rates):>7.1%} "
          f"{np.mean(cr_full_s):>7.4f} {np.mean(cr_full_u):>7.4f} {np.mean(cr_fallb):>7.4f}")


if __name__ == "__main__":
    main()
