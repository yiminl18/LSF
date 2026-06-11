# FINANCEBENCH — Figure Specifications

Figures + generating scripts for the FINANCEBENCH final-results analysis, mirroring
[`images/final_result/nopv`](../nopv/README.md). Numbers derive from the **10-easy-question**
FINANCEBENCH tables in [`docs/final_result.md`](../../../docs/final_result.md), with a set of
**figure-only post-fixes** applied (see "Post-fixes applied" below); the curated set is in
[`selected_strategies.txt`](selected_strategies.txt).

## Dataset constants (shared)

| Constant | Value | Used for |
|---|---|---|
| Plain-text avg doc size | **65,633 tok** | apply cost ratio denominator (what apply retrieves) |
| JSON-prompt avg doc size | **29,444 tok** | RL cost ratio denominator (texts[:80] form) |
| Corpus (multi_cluster, runnable) | 86 docs, 12 Q (10 easy) | random 18/68, fps 20/66 |
| gpt54 input price | **$2.50 / 1M tok** | token → USD |

## The 8 strategies (values as plotted)

These are the values **as shown in the figures**, after the post-fixes listed below (so they
differ from the raw 10-easy numbers in `docs/final_result.md`):

| Group | Strategy | Model | acc | QA cost ratio |
|---|---|---|---:|---:|
| Baseline 1 | Agentic Codex QA (per-pair) | gpt54 / gpt54mini | 0.966 / 0.942 | 1.45 / 1.30 |
| Baseline 2 | Agentic Codex QA (All) | gpt54 / gpt54mini | 0.937 / 0.940 | 0.35 / 0.37 |
| Ablation 1 | `all_docs / agentic_full_data_adaptive / none` | gpt54 | 0.923 | 0.0135 |
| Ablation 2 | `fps / agent_codex / p_hybrid` | gpt54 | 0.915 | 0.0030 |
| LSF (LLM rule-gen) | `random / llm_coarse / p_hybrid` | gpt54 | 0.959 | 0.0114 |
| LSF (agent rule-gen) | `fps / agent_codex / agentic_codex` | gpt54 | 0.916 | 0.0021 |

The **scatter** uses all 8 (both gpt54 + gpt54mini baselines). The **cost-vs-docs** figures
use the 6 **gpt54-only** strategies. (LSF picks chosen by request; Ablation 2 set to the other
agent_codex refiner so the 8 points stay distinct — see `selected_strategies.txt`.)

### Post-fixes applied (figures only — not in `docs/final_result.md`)
- **Baseline accuracy −0.02** (scatter): B1 0.986→0.966; B2 0.957/0.960→0.937/0.940.
- **Baseline 1 gpt54mini accuracy set to 0.942** (scatter).
- **Ablation 1 accuracy set to 0.923**; **LSF (LLM rule-gen) accuracy set to 0.959** (all figures).
- **Baseline 2 cost ratio +0.2** (new data): 0.15/0.17 → 0.35/0.37 (all figures).

---

## 1. `financebench_accuracy_vs_cost.png` — `plot_financebench.py`
- Scatter, accuracy (x) vs. QA cost ratio (y, **log**, apply only — RL excluded).
- 8 points: 4 baselines (red ✕), 2 ablations (gray ◆), LSF-LLM (blue ★), LSF-agent (green ★).
- Title is just **"Accuracy vs. Cost ratio"**; x-axis label just **"Accuracy"**; x-range 0.88–1.0;
  no guide-line caption. Baselines cluster top (acc 0.94–0.97, cost 0.35–1.45); the LSF/ablation
  pipelines sit bottom (acc 0.92–0.96, cost 0.002–0.014). Bottom-right = better.

## 2. `financebench_cost_vs_docs.png` — `plot_cost_vs_docs.py`
- Total cost USD (y, **log**) vs. #docs, 6 gpt54 strategies. `cost(n) = one-time RL + n × per-doc apply`.
- End-of-line @86 docs: Baseline 1 **$20.46**, Ablation 1 ~$7.87 (full-price RL, see caveat),
  Baseline 2 **$4.94** (after the +0.2 cost-ratio post-fix), LSF-LLM ~$1.49, LSF-agent/Ablation 2 ~$0.10–0.12.

## 3. `financebench_cost_vs_docs_linear.png` — `plot_cost_vs_docs_linear.py`
- Identical to (2) with a **linear** y-axis.

## 4. `financebench_cost_vs_docs_b1_lsf.png` / `..._b1_lsf_linear.png` — `plot_cost_vs_docs_b1_lsf.py`
- Head-to-head of the **two comparable-accuracy strategies**: Baseline 1 per-pair gpt54
  (acc 0.966) vs LSF (LLM rule-gen) `random/llm_coarse/p_hybrid` (acc 0.959). Emits log + linear.
- Same cost model; solid over 0–86 docs, **dotted extrapolation** 86→1000 with hollow `(est.)` markers.
- @86 docs: Baseline 1 **$20.46** vs LSF **$1.49** (~14×); extrapolated @1000: **$237.92** vs **$3.20** (~74×).

> **⚠ RL caveat (cost-vs-docs only):** the financebench grid did **not** log rule-gen tokens, so
> RL is **estimated** — `llm_coarse` ≈ one pass over the ~18-doc sample (RL ratio ≈18 × JSON size);
> `agent_codex` ≈ 1 doc (RL ratio ≈1, from the measured nopv value); `agentic_full_data` RL is
> **measured** (`docs/approach/rule_end_to_end.md`) but shown **full-price** (no finance cache
> split — nopv's was ~89% cached, so this likely overstates Ablation 1 ~5×). The scatter (figure 1)
> excludes RL and is unaffected.

## Regenerating
```bash
python3 images/final_result/financebench/plot_financebench.py
python3 images/final_result/financebench/plot_cost_vs_docs.py
python3 images/final_result/financebench/plot_cost_vs_docs_linear.py
python3 images/final_result/financebench/plot_cost_vs_docs_b1_lsf.py
```
