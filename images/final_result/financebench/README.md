# FINANCEBENCH — Figure Specifications

Figures + generating scripts for the FINANCEBENCH final-results analysis, mirroring
[`images/final_result/nopv`](../nopv/README.md). All numbers are the **10-easy-question**
values from the FINANCEBENCH tables in [`docs/final_result.md`](../../../docs/final_result.md);
the curated set is in [`selected_strategies.txt`](selected_strategies.txt).

## Dataset constants (shared)

| Constant | Value | Used for |
|---|---|---|
| Plain-text avg doc size | **65,633 tok** | apply cost ratio denominator (what apply retrieves) |
| JSON-prompt avg doc size | **29,444 tok** | RL cost ratio denominator (texts[:80] form) |
| Corpus (multi_cluster, runnable) | 86 docs, 12 Q (10 easy) | random 18/68, fps 20/66 |
| gpt54 input price | **$2.50 / 1M tok** | token → USD |

## The 8 strategies (10 easy questions)

| Group | Strategy | Model | acc |
|---|---|---|---:|
| Baseline 1 | Agentic Codex QA (per-pair) | gpt54 / gpt54mini | 0.986 / 0.976 |
| Baseline 2 | Agentic Codex QA (All) | gpt54 / gpt54mini | 0.957 / 0.960 |
| Ablation 1 | `all_docs / agentic_full_data_adaptive / none` | gpt54 | 0.967 |
| Ablation 2 | `fps / agent_codex / p_hybrid` | gpt54 | 0.915 |
| LSF (LLM rule-gen) | `random / llm_coarse / p_hybrid` | gpt54 | 0.953 |
| LSF (agent rule-gen) | `fps / agent_codex / agentic_codex` | gpt54 | 0.916 |

The **scatter** uses all 8 (both gpt54 + gpt54mini baselines). The **cost-vs-docs** figures
use the 6 **gpt54-only** strategies. (LSF picks chosen by request; Ablation 2 set to the other
agent_codex refiner so the 8 points stay distinct — see `selected_strategies.txt`.)

---

## 1. `financebench_accuracy_vs_cost.png` — `plot_financebench.py`
- Scatter, accuracy (x) vs. QA cost ratio (y, **log**, apply only — RL excluded).
- 8 points: 4 baselines (red ✕), 2 ablations (gray ◆), LSF-LLM (blue ★), LSF-agent (green ★).
- On easy-10 everything bunches at 0.91–0.99 acc; cost spans ~0.002–1.45 (bottom-right = better).

## 2. `financebench_cost_vs_docs.png` — `plot_cost_vs_docs.py`
- Total cost USD (y, **log**) vs. #docs, 6 gpt54 strategies. `cost(n) = one-time RL + n × per-doc apply`.
- End-of-line @86 docs: Baseline 1 **$20.46**, Ablation 1 ~$7.87 (full-price RL, see caveat),
  Baseline 2 $1.49, LSF-LLM ~$1.45, LSF-agent/Ablation 2 ~$0.18.

## 3. `financebench_cost_vs_docs_linear.png` — `plot_cost_vs_docs_linear.py`
- Identical to (2) with a **linear** y-axis.

> **⚠ RL caveat (cost-vs-docs only):** the financebench grid did **not** log rule-gen tokens, so
> RL is **estimated** — `llm_coarse` ≈ one pass over the ~18-doc sample (RL ratio ≈18 × JSON size);
> `agent_codex` ≈ 1 doc (RL ratio ≈1, from the measured nopv value); `agentic_full_data` RL is
> **measured** (`docs/approach/rule_end_to_end.md`) but shown **full-price** (no finance cache
> split — nopv's was ~89% cached, so this likely overstates Ablation 1 ~5×). The scatter (figure 1)
> excludes RL and is unaffected. The `_b1_lsf` head-to-head figures are intentionally **not** produced.

## Regenerating
```bash
python3 images/final_result/financebench/plot_financebench.py
python3 images/final_result/financebench/plot_cost_vs_docs.py
python3 images/final_result/financebench/plot_cost_vs_docs_linear.py
```
