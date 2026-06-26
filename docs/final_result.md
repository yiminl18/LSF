# Final Results — Baselines vs. LSF Pipeline

End-to-end comparison of the no-rule **baselines** (`docs/baseline/baseline.md`) against the
full **LSF rule pipelines** (`docs/pipeline.md`), one section per dataset.

## How to read the table

Each row is one strategy. Columns are split into two parts:

**Part 1 — Rule learning** (one-time, offline cost of producing the rule set)
- **RL cost ratio** = mean over queries of `(tokens used to generate that query's rules) /
  (average doc size in the representation fed to the generator)`. Reads as *"generation
  processed ~N sampled documents' worth of tokens per query."* Baselines learn no rules → **N/A**.
  - The denominator matches the generator's input representation, so numerator and denominator
    use the **same** units. `llm_coarse` feeds docs as indented JSON of their spans
    (`json.dumps(texts[:80], indent=2)`, nopv avg ≈ **19,761 tok/doc**, ~6.5× the plain-text
    size because of repeated field names + `indent=2` whitespace). `agent_codex` reads raw doc
    files selectively via Codex tools. So `llm_coarse ≈ 20` means *"one pass over the 20-doc
    sample"*; `agent_codex ≈ 1.3` means *"reads ~1 doc's worth, selectively."*
  *(Counts rule **generation** tokens only — Stage 2; both generators are single-pass — one
  LLM call / one Codex session per query. Refine/precompute tokens for the Pareto refiners are
  extra and not included; the generation pool is shared across the three refiners of a given
  `sampling/rule_gen` pair, so they share this value.)*

**Part 2 — Question answering** (per `(query, doc)` pair, at deployment)
- **QA accuracy**
  - *Pipelines:* combined over both pools — `acc = (n·sAcc + m·uAcc) / (n + m)`, where
    `n` = #docs in the sampling pool (sAcc) and `m` = #docs in the held-out pool (uAcc).
    For nopv `n ≈ 20`, `m ≈ 222` (242 docs total).
  - *Baselines:* fraction correct over all completed `(question, doc)` pairs.
- **QA cost ratio** — token cost of answering **one `(query, doc)` pair**, as a fraction of
  the document's size. Same unit for both:
  - *Baselines:* `input_tokens / doc_tokens` (the agent reads the whole doc + carries the
    Codex session overhead → ratio ≫ 1; the *All* variant amortizes the one-session fixed
    cost across all docs).
  - *Pipelines:* `retrieved_tokens / doc_tokens` — the **rule-application cost ratio**: only
    the rule-retrieved spans are fed to the answer model, so ratio ≪ 1. Combined over both
    pools with the same `(n, m)` weighting as accuracy.

> ⚠️ **Comparability caveat (nopv):** the baselines were measured on **24 questions × 50 docs**,
> while the pipelines were run on **12 questions × 242 docs**. The question sets and doc pools
> differ, so the **accuracy** columns are ballpark levels, not strict head-to-head deltas.
> (The cost ratios are per-pair and normalized by doc size, so they compare more directly.)

---

## NOPV

nopv avg doc size ≈ **3,025 tok** (plain text) / **19,761 tok** (JSON-prompt form), 12 queries,
250-doc corpus. Baselines: 24q × 50 docs (n = 1,200 pairs). Pipelines: 12q × 242 docs,
`apply=default`. RL ratio normalizes by JSON-form (matches generator input); QA cost ratio
normalizes by plain text (matches what apply retrieves).

| Strategy (sampling / rule_gen / refine) | Model | RL cost ratio | QA accuracy | QA cost ratio |
|---|---|---:|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | N/A | 0.922 | 33.89 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | N/A | 0.880 | 30.47 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | N/A | 0.867 | 1.67 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | N/A | 0.668 | 2.42 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | N/A‡ | 0.831 | 0.0672 |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | N/A‡ | 0.822 | 0.1153 |
| all_docs / agentic_full_data / none | gpt54mini | N/A‡ | 0.680 | 0.0464 |
| random / llm_coarse / p_mini | gpt54 | 20.6 | **0.935** | 0.4215 |
| fps / llm_coarse / p_mini | gpt54 | 18.2 | 0.932 | 0.3891 |
| fps / llm_coarse / agentic_codex | gpt54 | 18.2 | 0.931 | 0.2818 |
| fps / llm_coarse / p_hybrid | gpt54 | 18.2 | 0.929 | 0.2054 |
| fps / llm_coarse / p_hybrid / **descent** | gpt54 | 18.2 | 0.921 | 0.1919§ |
| random / llm_coarse / p_hybrid | gpt54 | 20.6 | 0.927 | 0.3630 |
| random / llm_coarse / agentic_codex | gpt54 | 20.6 | 0.901 | 0.2970 |
| fps / agent_codex / agentic_codex | gpt54 | 1.3† | 0.853 | 0.0394 |
| fps / agent_codex / p_hybrid | gpt54 | 1.3† | 0.853 | 0.0403 |
| fps / agent_codex / p_mini | gpt54 | 1.3† | 0.852 | 0.0402 |
| random / agent_codex / p_hybrid | gpt54 | 1.3 | 0.846 | 0.0387 |
| random / agent_codex / p_mini | gpt54 | 1.3 | 0.846 | 0.0387 |
| random / agent_codex / agentic_codex | gpt54 | 1.3 | 0.843 | 0.0373 |
| fps / **agent_codex_val** / none ¶ | gpt54 | ~250◊ | 0.906 | 0.0368 |

*sAcc/uAcc breakdown (pipelines, for reference):*

| sampling / rule_gen / refine | sAcc | uAcc | combined acc |
|---|---:|---:|---:|
| random / llm_coarse / p_mini | 0.912 | 0.937 | 0.935 |
| fps / llm_coarse / p_mini | 0.938 | 0.931 | 0.932 |
| fps / llm_coarse / agentic_codex | 0.950 | 0.929 | 0.931 |
| fps / llm_coarse / p_hybrid | 0.929 | 0.929 | 0.929 |
| fps / llm_coarse / p_hybrid / descent | 0.929 | 0.920 | 0.921 |
| random / llm_coarse / p_hybrid | 0.887 | 0.931 | 0.927 |
| random / llm_coarse / agentic_codex | 0.850 | 0.906 | 0.901 |
| fps / agent_codex / agentic_codex | 0.942 | 0.845 | 0.853 |
| fps / agent_codex / p_hybrid | 0.942 | 0.845 | 0.853 |
| fps / agent_codex / p_mini | 0.917 | 0.847 | 0.852 |
| random / agent_codex / p_hybrid | 0.925 | 0.839 | 0.846 |
| random / agent_codex / p_mini | 0.925 | 0.839 | 0.846 |
| random / agent_codex / agentic_codex | 0.917 | 0.836 | 0.843 |

† `fps/agent_codex` rule-generation tokens were **not recorded** (Codex subprocess token
usage wasn't captured for that run); the RL cost ratio shown reuses the `random/agent_codex`
generation total (≈307.6K tokens) as a proxy. All other RL ratios are from measured tokens.

‡ `agentic_full_data` = the *rule-end-to-end* approach (`src/baseline/agentic_rule_full_data.py`,
documented in `docs/approach/rule_end_to_end.md`): a Codex agent learns Python retrieval rules
from the **full 242-doc corpus** (no 20-doc sample), then rules are applied with `rule_apply_merge`
(gpt54 answers/judges), scored over **all 242 docs** (no sampled/unsampled split). `adaptive` =
`--adaptive-large-sample` (larger, spread working sample during rule-gen). RL cost ratio is **N/A**
because these runs read raw `.txt` docs on demand over the whole corpus, so the grid's
"sampled docs' worth of JSON-prompt tokens per query" denominator doesn't apply; the cost ratio
shown is the **apply (QA) cost only**.

§ `/ descent` = the same refined rules as the `fps / llm_coarse / p_hybrid` row, but applied with
the **Cost-Descent** strategy (`src/rule_apply/descent.py`, see `docs/approach/rule_apply_descent.md`)
instead of `default`: rules are cost-sorted and the gpt54 context is halved toward the cheapest
rules while a gpt54mini gate still finds the answer, with full-pool fallback. **Cost counts gpt54
tokens only** (gpt54mini gate calls are logged but excluded). On nopv it shrank the gpt54 context to
**~36% of the refined set on average** (9.8% of docs answered from the single cheapest rule; 32.1%
hit fallback), trading **~0.8 pt accuracy** (0.929→0.921, all on the unsampled split) for **~6%
lower cost** (0.205→0.192). RL cost ratio is unchanged (same rule generation).

¶ `agent_codex_val` = **validation-guarded** Codex rule-gen (trains on the 20 sampled docs,
then validates on a held-out 20-doc set and broadens overfit rules; see the dedicated
subsection below). This row uses **no refine** and **`merge`** apply (not `default` like the
rest of the table), and reports combined accuracy (sampled 0.954 / unsampled 0.902). The
`~250◊` RL ratio reflects the heavy multi-turn agent session (59.2M gen input tokens,
~$80.7 total); ◊ marks it approximate (per-turn usage includes resent/cached context — treat
as an upper bound). It improves unsampled accuracy +5.7 pt over `agent_codex` (0.845→0.902)
but at ~190× the rule-learning cost, so it is **not** Pareto-optimal.

### Analysis

**1. The pipeline's whole value is in the QA cost ratio.** Every pipeline answers from
**rule-retrieved spans**, so its per-pair cost ratio is **far below 1** (0.037–0.42 of the
document), whereas the baselines feed the model the **whole document plus the Codex session
overhead** — 1.67–2.42 for the amortized *All* reader, and **30–34** for the per-pair agent.
So even the most expensive pipeline (`random/llm_coarse/p_mini`, 0.42) is **~4× cheaper per
pair** than the cheapest baseline (Baseline 2, 1.67) and **~80× cheaper** than Baseline 1.

**2. Accuracy: pipelines match or beat the baselines.** The 6 `llm_coarse` pipelines reach
**0.90–0.935 combined accuracy**, at or above the strongest baseline (Baseline 1 gpt54,
0.922), and well above the realistic cost-floor baseline (Baseline 2 gpt54, 0.867; mini
collapses to 0.668). `random/llm_coarse/p_mini` (0.935) is the top scorer overall.

**3. But accuracy costs rule-learning tokens.** The split is at rule-gen, not apply. Both
generators are single-pass, but they read different amounts of the sample: `llm_coarse`
ingests the **whole 20-doc sample once** (RL ≈ **18–21** docs' worth) and lands at 0.90–0.935;
`agent_codex` reads only **~1 doc's worth selectively** (RL ≈ **1.3**, ~15× cheaper) and lands
~8 pts lower at 0.843–0.853. So `llm_coarse`'s accuracy edge is bought by feeding the model
the entire sample, while `agent_codex` is far leaner but weaker. There is no strategy that is
both top-accuracy and cheap-to-learn (unlike court, where `agent_codex` Pareto-dominated).

**4. Refiner choice moves the QA cost ratio, barely touches accuracy.** Within `llm_coarse`,
all three refiners are within ~3 accuracy pts, but their apply cost ratio differs ~2×:
`p_hybrid` retrieves the least (0.205 fps / 0.363 random), `agentic_codex` is middling
(0.282 / 0.297), `p_mini` the most (0.389 / 0.421). So `fps/llm_coarse/p_hybrid`
(0.929 acc @ **0.205**) is the best accuracy-per-cost point in the high tier — same accuracy
as `fps/llm_coarse/p_mini` (0.932 @ 0.389) for **~1.9× less apply cost**. For `agent_codex`
the refiners are effectively tied on both axes (acc 0.843–0.853, cost 0.037–0.040).

**Bottom line for nopv:**
- **Max accuracy:** `random/llm_coarse/p_mini` — 0.935 acc @ 0.42 cost ratio (top accuracy, highest pipeline cost, still ≫ cheaper than any baseline).
- **Best accuracy-per-cost:** `fps/llm_coarse/p_hybrid` — 0.929 acc @ 0.205, the cheapest of the high-accuracy tier.
- **Cheapest overall:** `random/agent_codex/agentic_codex` — 0.843 acc @ 0.037 cost ratio, ~15× cheaper to learn than any `llm_coarse` and still above Baseline 2 mini.

### Validation-guarded rule-gen (`agent_codex_val`) — new experiment (2026-06-13)

A variant of `agent_codex` rule generation with a **generalization guard**: the Codex
agent trains rules on the 20 sampled docs as usual, then validates them on a **held-out
20-doc set** carved from the unsampled pool (never seen during rule design), broadening
overfit rules (≤ 3 passes) until validation accuracy holds within 5 pts of sampled. Run
with **no refine** and **`merge`** apply (generated rules applied directly — no gpt54mini
gate, no fallback). Reported with a three-way split (option-3): `sampled` / `val`
(held-out) / `clean-test` (unsampled **minus** val, no leakage) / `full-unsampled`
(legacy, comparable to the rows above). fps, gpt54, all 12 questions, 222-doc unsampled.

| metric | value |
|---|---:|
| sampled accuracy | 0.954 |
| validation accuracy (held-out 20) | 0.917 |
| clean-test accuracy (no leakage) | 0.900 |
| full-unsampled accuracy | 0.902 |
| QA cost ratio (`merge` apply) | 0.037 |
| rule-gen tokens — codex agent | 59.2M in / 0.65M out |
| rule-gen tokens — LLM verification | 0.17M in / 4K out |
| rule-gen cost (gpt-5.4 rates) | **~$80.7** |
| RL cost ratio (≈ gen-input ÷ 1-doc JSON form) | **~250** ◊ |

**Comparison to the current nopv results:**

| approach | sAcc | uAcc | sampled→uAcc gap | QA cost | RL ratio | gen cost |
|---|---:|---:|---:|---:|---:|---:|
| `fps/agent_codex/agentic_codex` (prior agentic) | 0.942 | 0.845 | 9.7 pt | 0.039 | 1.3 | ~$0.4 |
| **`fps/agent_codex_val/merge` (new)** | 0.954 | **0.902** | **5.2 pt** | 0.037 | ~250 | **~$80.7** |
| `fps/llm_coarse/agentic_codex` (best non-agentic gen) | 0.950 | **0.929** | 2.1 pt | 0.282 | 18.2 | — |

**Takeaways:**
- **Accuracy + generalization improved over `agent_codex`:** unsampled **0.845 → 0.902
  (+5.7 pt)**, and the sampled→unsampled gap nearly halved (9.7 → 5.2 pt). The validation
  guard does reduce overfitting, as intended; apply cost is unchanged (~0.037).
- **But it is not Pareto-optimal.** It still trails `llm_coarse` on accuracy (0.902 vs
  0.929) while costing **~190× more to learn** than `agent_codex` (59.2M vs ~0.3M gen
  tokens; **~$80.7 vs ~$0.4**) and ~14× the rule-learning of `llm_coarse` (RL ≈ 250 vs
  18). The gain (+5.7 pt over `agent_codex`) does not justify the ~200× rule-gen cost.
- **Cost is highly question-dependent:** easy cover-page questions used ~2M gen tokens;
  the hardest (list all 49 CFR sections) used **12.2M**. Most validation was done with the
  substring proxy / the agent's own reasoning; only the counting/listing questions
  triggered separate gpt54 judge calls (verification tokens, 0.17M total).

> ◊ The RL cost ratio for `agent_codex_val` is approximate: codex `exec --json` reports
> per-turn usage that includes resent (partly cached) context, so the 59.2M input is
> inflated by multi-turn accumulation. Treat ~250 as an order-of-magnitude upper bound on
> rule-learning cost, not a precise figure. (Result files:
> `results/nopv/grid/{apply,rule_gen}/fps/agent_codex_val_gpt54/...`.)

### NOPV — normalized (single cost ratio)

Both numeric columns are **weighted averages over all 242 docs** (n = 20 sampled, m = 222
unsampled). How each column is computed:

**Strategy** — `sampling / rule_gen / refine` (pipelines); apply is `default` throughout.
Baselines have no pipeline stages.

**Model** — answer/judge model (`gpt54` or `gpt54mini`). Pipelines use `gpt54` for apply.

**Accuracy** = `(n·sAcc + m·uAcc) / (n + m)` = `(20·sAcc + 222·uAcc) / 242`.
- `sAcc` = mean apply accuracy over the 20 sampled docs (mean over the 12 questions).
- `uAcc` = mean apply accuracy over the 222 held-out docs (mean over the 12 questions).
- *Baselines:* fraction of correct `(question, doc)` pairs over all evaluated pairs (no
  sampling split — there is no rule learning).

**Cost ratio** = `(n·sampled_cr + m·unsampled_cr) / (n + m)` = `(20·sampled_cr + 222·unsampled_cr) / 242`.
- `sampled_cr = RL_cost_ratio / 20` — the rule-learning cost amortized over the 20 sampled docs.
  - `RL_cost_ratio` = mean over the 12 queries of `(rule-gen tokens for that query) / (avg JSON-prompt doc size)`.
    Rule-gen tokens = `input + output` of the single rule-generation call/session per query
    (Stage 2 only; refine/precompute excluded). The denominator is the doc size in the *same*
    representation the generator reads — `json.dumps(texts[:80], indent=2)`, nopv avg ≈ 19,761 tok —
    so `RL_cost_ratio` reads as "sampled docs' worth of tokens processed per query"
    (`llm_coarse` ≈ 18–21, `agent_codex` ≈ 1.3).
- `unsampled_cr` = mean apply cost ratio over the 222 held-out docs = mean of
  `retrieved_tokens / doc_tokens` per `(query, held-out doc)` pair (plain-text doc tokens).
- *Baselines:* learn no rules, so cost ratio is just their per-pair QA token ratio
  `input_tokens / doc_tokens` (Baseline 1 per-pair; Baseline 2 amortizes its one-session fixed
  cost across all docs) — no sampling split.

> Note: the two columns share the same 20/222 weighting, but the **sampled-doc term differs by
> column** — accuracy counts the sampled docs' *answering* (`sAcc`), while cost counts their
> *rule-learning* (`RL/20`), not their apply cost. Also, `sampled_cr` is normalized by the
> JSON-prompt doc size (what gen reads) and `unsampled_cr` by the plain-text doc size (what
> apply retrieves) — different denominators folded into one ratio.

| Strategy (sampling / rule_gen / refine) | Model | Accuracy | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | 0.922 | 33.89 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.880 | 30.47 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.867 | 1.67 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.668 | 2.42 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | 0.831 | 0.0672‡ |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | 0.822 | 0.1153‡ |
| all_docs / agentic_full_data / none | gpt54mini | 0.680 | 0.0464‡ |
| random / llm_coarse / p_mini | gpt54 | **0.935** | 0.4743 |
| fps / llm_coarse / p_mini | gpt54 | 0.932 | 0.4310 |
| fps / llm_coarse / agentic_codex | gpt54 | 0.931 | 0.3327 |
| fps / llm_coarse / p_hybrid | gpt54 | 0.929 | 0.2620 |
| fps / llm_coarse / p_hybrid / **descent** | gpt54 | 0.921 | 0.2499§ |
| random / llm_coarse / p_hybrid | gpt54 | 0.927 | 0.4204 |
| random / llm_coarse / agentic_codex | gpt54 | 0.901 | 0.3603 |
| fps / agent_codex / p_hybrid | gpt54 | 0.853 | 0.0422 |
| fps / agent_codex / agentic_codex | gpt54 | 0.853 | 0.0414 |
| fps / agent_codex / p_mini | gpt54 | 0.852 | 0.0422 |
| random / agent_codex / p_mini | gpt54 | 0.846 | 0.0410 |
| random / agent_codex / p_hybrid | gpt54 | 0.846 | 0.0410 |
| random / agent_codex / agentic_codex | gpt54 | 0.843 | 0.0397 |

‡ For the `agentic_full_data` (rule-end-to-end) rows the cost ratio is the **apply (QA) cost
only** — there is no 20-doc sample, so no `RL/20` term is folded in. See the ‡ footnote under the
main NOPV table for the full definition.

§ `/ descent` = `fps / llm_coarse / p_hybrid`'s refined rules applied with the **Cost-Descent**
strategy (gpt54-only cost; see the § footnote under the main NOPV table). Normalized cost folds in
the same `RL/20` term as the `default` row (`18.2/20`) over the 20 sampled docs, plus descent's
lower unsampled apply cost (0.190 vs 0.204) over the 222 held-out docs: `(20·0.91 + 222·0.190)/242
= 0.2499`.

---

## COURT

court avg doc size ≈ **9,912 tok** (plain text) / **27,615 tok** (JSON-prompt form, ~2.8×
inflation), 13 queries, 320-doc corpus, split **n = 20 sampled / m = 274 unsampled / 294 total**.
Baselines: Agentic Codex QA per-pair = 13q × 50 docs (n = 650); Agentic Codex QA All = 10q × 294
docs (n = 2,940). Column definitions identical to the NOPV section above (RL ratio normalizes by
JSON-form, QA cost ratio by plain text).

| Strategy (sampling / rule_gen / refine) | Model | RL cost ratio | QA accuracy | QA cost ratio |
|---|---|---:|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | N/A | 0.918 | 33.32 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | N/A | 0.906 | 30.80 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | N/A | 0.880 | 1.33 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | N/A | 0.846 | 1.61 |
| fps / agent_codex / agentic_codex | gpt54 | 0.21 | **0.925** | 0.0058 |
| fps / agent_codex / p_mini | gpt54 | 0.21 | 0.925 | 0.0063 |
| fps / agent_codex / p_hybrid | gpt54 | 0.21 | 0.925 | 0.0063 |
| fps / agent_codex / p_hybrid / **descent** | gpt54 | 0.21 | 0.925 | 0.0062§ |
| random / agent_codex / p_hybrid | gpt54 | 0.22† | 0.911 | 0.0054 |
| random / agent_codex / p_mini | gpt54 | 0.22† | 0.910 | 0.0054 |
| random / agent_codex / agentic_codex | gpt54 | 0.22† | 0.909 | 0.0047 |
| fps / llm_coarse / p_mini | gpt54 | 13.5 | 0.874 | 0.0778 |
| fps / llm_coarse / p_hybrid | gpt54 | 13.5 | 0.873 | 0.0523 |
| fps / llm_coarse / agentic_codex | gpt54 | 13.5 | 0.867 | 0.0556 |
| random / llm_coarse / p_mini | gpt54 | 19.8† | 0.846 | 0.0308 |
| random / llm_coarse / p_hybrid | gpt54 | 19.8† | 0.846 | 0.0258 |
| random / llm_coarse / agentic_codex | gpt54 | 19.8† | 0.845 | 0.0174 |
| full_data / agentic_rule (gpt54, adaptive) / merge | gpt54 | N/A‡ | **0.807** | 0.0325 |
| full_data / agentic_rule (gpt54mini, adaptive) / merge | gpt54mini | N/A‡ | 0.751 | 0.0332 |
| full_data / agentic_rule (gpt54mini, baseline) / merge | gpt54mini | N/A‡ | 0.735 | 0.0295 |

*sAcc/uAcc breakdown (pipelines):*

| sampling / rule_gen / refine | sAcc | uAcc | combined acc |
|---|---:|---:|---:|
| fps / agent_codex / agentic_codex | 0.965 | 0.922 | 0.925 |
| fps / agent_codex / p_mini | 0.965 | 0.922 | 0.925 |
| fps / agent_codex / p_hybrid | 0.965 | 0.922 | 0.925 |
| fps / agent_codex / p_hybrid / descent | 0.969 | 0.922 | 0.925 |
| random / agent_codex / p_hybrid | 0.973 | 0.907 | 0.911 |
| random / agent_codex / p_mini | 0.969 | 0.906 | 0.910 |
| random / agent_codex / agentic_codex | 0.973 | 0.904 | 0.909 |
| fps / llm_coarse / p_mini | 0.842 | 0.876 | 0.874 |
| fps / llm_coarse / p_hybrid | 0.846 | 0.875 | 0.873 |
| fps / llm_coarse / agentic_codex | 0.854 | 0.868 | 0.867 |
| random / llm_coarse / p_mini | 0.865 | 0.845 | 0.846 |
| random / llm_coarse / p_hybrid | 0.865 | 0.845 | 0.846 |
| random / llm_coarse / agentic_codex | 0.854 | 0.845 | 0.845 |

† RL averaged over the queries whose gen tokens were recorded (Codex-subprocess token capture
missed `random/agent_codex` 11/13 and `random/llm_coarse` 12/13; `fps` runs recorded all 13).

‡ `full_data / agentic_rule` rows are from `docs/approach/rule_end_to_end.md` — full-data rule
generation over all 294 docs (no sampled/unsampled split; apply uses `rule_apply_merge` with
`gpt54` for gen+judge). RL not normalized as a single ratio here; raw rule-gen tokens (in/out):
gpt54-adaptive 40,650,849 / 410,238; gpt54mini-adaptive 62,487,212 / 746,101; gpt54mini-baseline
35,095,343 / 455,270.

§ `/ descent` = the same refined rules as the `fps / agent_codex / p_hybrid` row (the
**lsf_agent_rule_gen** combo), but applied with the **Cost-Descent** strategy
(`src/rule_apply/descent.py`, see `docs/approach/rule_apply_descent.md`) instead of `default`: rules
are cost-sorted and the gpt54 context is halved toward the cheapest rules while a gpt54mini gate
still finds the answer, with full-pool fallback. **Cost counts gpt54 tokens only** (gpt54mini gate
calls are logged but excluded). Unlike nopv, descent here is a near-pure win — accuracy is flat
(sAcc 0.965→0.969, uAcc unchanged at 0.922; combined 0.925 either way) for **~2.3% lower apply
cost** (per-doc unsampled cost ratio 0.00518→0.00506). Gains are modest because the refined
p_hybrid set is already lean (3–6 rules/question, little to prune) and **28.2% of unsampled docs
(44.6% sampled) hit the full-pool fallback**. Per-doc shrink depth (`final_k/final_n`) isn't
persisted by the eval wrapper, so fallback rate is the shrink proxy. RL cost ratio is unchanged
(same rule generation). Run: `results/court/grid/apply/fps/agent_codex_gpt54/p_hybrid/descent`.

### Analysis

**`agent_codex` Pareto-dominates on court** — the opposite of nopv. All 6 `agent_codex`
pipelines are simultaneously **more accurate** (0.909–0.925) **and ~15× cheaper to apply**
(QA cost 0.005–0.006) than the 6 `llm_coarse` pipelines (0.845–0.874 acc, 0.017–0.078 cost),
and they're also **~15–90× cheaper to learn** (RL ≈ 0.2 vs 13–20). On court, Codex's selective
reading finds tight, accurate rules; `llm_coarse`'s broad JSON dump generates weaker, costlier
rules.

**Pipelines beat the best baseline on both axes.** The top pipeline `fps/agent_codex/*`
(**0.925**) exceeds the strongest baseline (Baseline 1 gpt54, 0.918) while costing **0.006 vs
33.32** — i.e. ~0.017 % of the baseline's QA cost (~5,700× cheaper) at *higher* accuracy.

### COURT — normalized (single cost ratio)

Both columns are weighted averages over all 294 docs (n = 20 sampled, m = 274 unsampled),
computed exactly as in the NOPV normalized section: `Accuracy = (20·sAcc + 274·uAcc)/294`;
`Cost ratio = (20·(RL/20) + 274·unsampled_cr)/294`.

| Strategy (sampling / rule_gen / refine) | Model | Accuracy | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | 0.918 | 33.32 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.906 | 30.80 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.880 | 1.33 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.846 | 1.61 |
| fps / agent_codex / agentic_codex | gpt54 | **0.925** | 0.0051 |
| fps / agent_codex / p_mini | gpt54 | 0.925 | 0.0055 |
| fps / agent_codex / p_hybrid | gpt54 | 0.925 | 0.0055 |
| fps / agent_codex / p_hybrid / **descent** | gpt54 | 0.925 | 0.0054§ |
| random / agent_codex / p_hybrid | gpt54 | 0.911 | 0.0059 |
| random / agent_codex / p_mini | gpt54 | 0.910 | 0.0059 |
| random / agent_codex / agentic_codex | gpt54 | 0.909 | 0.0052 |
| fps / llm_coarse / p_mini | gpt54 | 0.874 | 0.1162 |
| fps / llm_coarse / p_hybrid | gpt54 | 0.873 | 0.0919 |
| fps / llm_coarse / agentic_codex | gpt54 | 0.867 | 0.0953 |
| random / llm_coarse / p_mini | gpt54 | 0.846 | 0.0964 |
| random / llm_coarse / p_hybrid | gpt54 | 0.846 | 0.0917 |
| random / llm_coarse / agentic_codex | gpt54 | 0.845 | 0.0840 |
| full_data / agentic_rule (gpt54, adaptive) / merge | gpt54 | **0.807** | 0.0325 |
| full_data / agentic_rule (gpt54mini, adaptive) / merge | gpt54mini | 0.751 | 0.0332 |
| full_data / agentic_rule (gpt54mini, baseline) / merge | gpt54mini | 0.735 | 0.0295 |

(`full_data / agentic_rule` rows from `docs/approach/rule_end_to_end.md` — full-data rule-gen over
all 294 docs, no sampled/unsampled split, so Accuracy/Cost ratio are the direct all-docs values.)

§ `/ descent` normalized cost folds in the same `RL/20` term as its `default` row (`0.21/20`) over
the 20 sampled docs, plus descent's lower unsampled apply cost (0.00506 vs 0.00518) over the 274
held-out docs: `(20·(0.21/20) + 274·0.00506)/294 = 0.0054`. See the § footnote under the main court
table for the strategy definition.

---

## FINANCEBENCH

> ⚠️ **No 12-combo grid was run for financebench.** The pipeline rows below are *not* the
> `sampling × rule_gen × refine` grid used for court/nopv. They are: (a) **`agentic_full_data`**
> runs — the *rule-end-to-end* approach (`src/baseline/agentic_rule_full_data.py`) evaluated
> over **all 100 docs**; and (b) one **legacy `multi_clusters` llm_coarse** pipeline. Two
> caveats specific to finance:
> 1. **QA cost only.** The `agentic_full_data` runs *do* record rule-gen tokens, but the legacy
>    pipeline does not — so for one comparable axis the cost ratio shown is the **apply (QA) cost
>    only** (RL not folded in); the main and normalized cost columns therefore coincide.
> 2. **Finance docs are huge** (avg ≈ **68,116 plain tokens**, 142-doc corpus) — so the
>    baselines' `input/doc` ratios are *low* here (0.15–1.45), unlike court/nopv where small
>    docs inflated them to 30+. Comparisons read differently as a result.

Evaluation splits: `agentic_full_data` rows are scored over **all 100 docs** (`all_docs`);
accuracy = mean over 12 questions of per-doc correctness, cost ratio = mean apply
`retrieved/doc`. The legacy `multi_clusters` pipeline uses an 18 sampled / 68 unsampled split,
accuracy = `(ns·sAcc + nu·uAcc)/(ns+nu)`. Baselines: per-pair = 39–59 docs/q; All = 10q × 59 docs.

| Strategy (cluster / rule_gen / refine) | Model | RL cost ratio | Accuracy | QA cost ratio |
|---|---|---:|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | N/A | 0.931 | 1.45 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | N/A | 0.878 | 1.30 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | N/A | 0.861 | 0.15 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | N/A | 0.820 | 0.17 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | N/A† | **0.923** | 0.0135 |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | N/A† | 0.879 | 0.0139 |
| all_docs / agentic_full_data / none | gpt54mini | N/A† | 0.854 | 0.0141 |
| multi_clusters / llm_coarse / raw | gpt54 | N/A‡ | 0.846 | 0.0829 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | N/A‡ | 0.848§ | **0.0090** |

*sAcc/uAcc breakdown (pipelines):*

| cluster / rule_gen / refine | Model | ns/nu | sAcc | uAcc | combined acc |
|---|---|---|---:|---:|---:|
| multi_clusters / llm_coarse / raw | gpt54 | 18/68 | 0.857 | 0.843 | 0.846 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | —/68 | — | 0.848 | 0.848§ |

(The `agentic_full_data` runs have no sampled/unsampled split — they are scored over all 100 docs.)

† `agentic_full_data` = the *rule-end-to-end* approach (`src/baseline/agentic_rule_full_data.py`):
a Codex agent learns Python retrieval rules from the corpus, then rules are applied with
`rule_apply_merge` (gpt54 answers/judges), scored over all 100 docs. `adaptive` =
`--adaptive-large-sample` (larger, spread working sample during rule-gen). Rule-gen tokens are
recorded but not folded into the cost ratio here.<br>
‡ Legacy run — rule-generation tokens were not recorded, so RL cost ratio is not computable.<br>
§ `agentic+fallback` = a *refinement* of the `multi_clusters/llm_coarse/raw` one-shot pool: an
agentic selector (**Claude Opus 4.7**) picks ~2.7 rules/question from the ~35-rule pool, then a
**gpt54mini gate** applies them with **full-pool fallback** on a miss (gpt54 answers/judges).
Evaluated on the **held-out 68 unsampled docs only** (no sampled split), all 12 questions; the
cost ratio is gpt54 input-per-doc / avg doc tokens. Over the 10 easier questions uAcc = 0.940
(the 2 hard ones — long-term debt 0.500, exhibit listing 0.279 — pull the 12-q mean to 0.848).

### Analysis

**Finance baselines are strong and (here) cheap.** Because finance docs are huge, the per-pair
Codex baseline reads ≈ one doc's worth (ratio 1.45) and the amortized *All* baseline drops to
0.15 — both far below the 30+ ratios seen on court/nopv. The best baseline (Codex per-pair
gpt54, **0.931**) sets a high accuracy bar.

**The `agentic_full_data` pipeline nearly matches the best baseline at ~100× lower cost.**
`agentic_full_data_adaptive` (gpt54) reaches **0.923** over all 100 docs at a **0.0135** cost
ratio — within ~1 pt of the strongest baseline (Codex per-pair gpt54, 0.931) but **~107×
cheaper** (0.0135 vs 1.45). On gpt54mini, the `adaptive` variant (0.879) beats the non-adaptive
run (0.854), and gpt54 generation beats gpt54mini. The legacy `multi_clusters/llm_coarse/raw` (gpt54)
reaches **0.846** at 0.083 — solid, but dominated by `agentic_full_data` on both axes
(lower accuracy *and* ~6× more expensive).

**Agentic selection makes the one-shot pool ~9× cheaper at the same accuracy.** Layering
`agentic+fallback` on the `multi_clusters/llm_coarse/raw` pool holds held-out accuracy flat
(uAcc 0.848 vs the raw pool's 0.843) while cutting the apply cost from **0.0829 → 0.0090** —
because the agentic selector trims ~35 rules to ~2.7 and the gpt54mini gate only escalates to
the full pool ~11% of the time. That 0.0090 cost is now the **cheapest pipeline on financebench**,
and roughly ties the `agentic_full_data` runs on cost while trailing them on accuracy
(0.848 vs 0.923). So the two strong options are: `agentic_full_data_adaptive` (gpt54) for **peak
accuracy** (0.923 @ 0.0135), or `agentic+fallback` for **lowest cost** (0.848 @ 0.0090). Note the
agentic+fallback *selection* step uses Claude Opus 4.7 (one-time), though all QA is gpt54/gpt54mini.

**Caveat:** with the 12-combo grid not run and heterogeneous evaluation splits (all_docs vs
18/68), these finance numbers are **not directly comparable** to the court/nopv grids — treat
them as the best available snapshot, not a like-for-like sweep. Running the proper 12-combo grid
would be needed for a clean comparison.

### FINANCEBENCH — normalized (single cost ratio)

Same columns as the other normalized tables. **RL is unavailable for these legacy runs**, so
the cost ratio is the **combined apply cost only** (no `RL/20` term folded in) — i.e. it equals
the QA cost ratio above.

| Strategy (cluster / rule_gen / refine) | Model | Accuracy | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | 0.931 | 1.45 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.878 | 1.30 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.861 | 0.15 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.820 | 0.17 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | **0.923** | 0.0135 |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | 0.879 | 0.0139 |
| all_docs / agentic_full_data / none | gpt54mini | 0.854 | 0.0141 |
| multi_clusters / llm_coarse / raw | gpt54 | 0.846 | 0.0829 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | 0.848§ | **0.0090** |

### FINANCEBENCH — 10 easy questions (baselines vs pipelines)

The 12-question set contains 2 structurally hard questions — **long-term debt** (uAcc ≈ 0.50;
numeric extraction across varied tables) and **exhibit / material-agreement listing** (uAcc ≈
0.28; free-form extraction across heterogeneous indices) — that drag every pipeline's mean down.
This table drops those 2 and reports the remaining **10 "easy" questions**. Pipeline eval scopes
are unchanged (†/‡/§). **Baselines were re-run on the 5 easy questions absent from their original
set** (`run_easy5_baselines.sh`, 50 docs with ground truth), then combined with their 5 shared
questions for a full easy-10 mean.

| Strategy (cluster / rule_gen / refine) | Model | Accuracy | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | **0.986** | 1.45 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.976 | 1.30 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | 0.967 | 0.0135 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.960 | 0.17 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.957¶ | 0.15 |
| multi_clusters / llm_coarse / raw | gpt54 | 0.943 | 0.0540 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | 0.940§ | **0.0090** |
| all_docs / agentic_full_data / none | gpt54mini | 0.935 | 0.0141 |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | 0.923 | 0.0140 |

¶ All-gpt54 `reporting period` was re-judged 0.00 → 1.00: codex answered the correct fiscal-year-end
date (e.g. "December 31, 2017") but the standard judge rejected all 50 for omitting the
"fiscal year ended" prefix; both a clarified LLM judge and a date-equivalence check score 50/50.
Without this fix All-gpt54 easy-10 reads 0.757.

**On the easy 10, baselines lead on accuracy but at 10–160× the cost.** The per-pair gpt54
baseline tops the table at **0.986**, but the best pipeline — `agentic_full_data_adaptive` (gpt54,
**0.967**) — sits *above* both "All" baselines (0.957–0.960) while costing **~11× less than "All"
gpt54 and ~107× less than per-pair gpt54**. The cheapest pipeline, `agentic+fallback`, holds
**0.940 at 0.0090** — within ~5 pts of the per-pair baseline at a tiny fraction of the cost. So
even on the questions most favorable to the baselines (easy cover-page facts), the accuracy gap is
small and the cost gap is enormous. The full-pool raw's per-split numbers here (sAcc 0.972 /
uAcc 0.935 @ 0.0356 / 0.0589) are the source of the `Table 5` screenshot. Note the gpt54mini
*pipeline* ordering flips vs the 12-question table: non-adaptive (0.935) edges adaptive (0.923) on
the easy subset, where the adaptive sampler's extra rules mainly helped the 2 hard questions.

### FINANCEBENCH — matched comparison (6 shared questions, baselines vs pipelines)

The baselines run on a **different** 10-question set than the LSF pipelines; only **6 questions
appear in both** — trading symbols, long-term debt, registrant name, exec-office address/ZIP,
telephone, state/jurisdiction. This table scores **every system on exactly those 6 questions**,
so accuracy is finally apples-to-apples (doc sets still differ: baselines ≈ 59 single-cluster
docs/q, pipelines 86–100 multi-cluster docs). Cost = each system's overall ratio (footnotes
†/‡/§ as above; baseline cost is full-doc and question-independent).

| Strategy (cluster / rule_gen / refine) | Model | Acc (6 shared Q) | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | **0.938** | 1.45 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.927 | 1.30 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.896 | 0.15 |
| all_docs / agentic_full_data_adaptive / none | gpt54 | 0.887 | 0.0135 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.879 | 0.17 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | 0.848§ | **0.0090** |
| multi_clusters / llm_coarse / raw | gpt54 | 0.839 | 0.0829 |
| all_docs / agentic_full_data_adaptive / none | gpt54mini | 0.835 | 0.0139 |
| all_docs / agentic_full_data / none | gpt54mini | 0.757 | 0.0141 |

**On matched questions, the baselines lead on accuracy — pipelines win only on cost.** The best
baseline (Codex per-pair gpt54) hits **0.938** vs the best pipeline's **0.887**
(`agentic_full_data_adaptive` gpt54). This is the honest read: the pipelines' headline 0.92+
elsewhere was lifted by easy questions (stock exchange, form type, reporting period) that the
baseline set doesn't contain, while these 6 shared questions include the hard ones (long-term
debt, trading symbols). The pipelines' real advantage is **cost** — `agentic+fallback` answers
the same 6 questions at **0.0090** (≈0.85 acc) vs the per-pair baseline's **1.45** (≈161× cheaper)
and even the amortized "All" baseline's 0.15 (≈17× cheaper), at 4–9 accuracy points lower.

### FINANCEBENCH — the proper 12-combo grid (all 12 complete)

The canonical `sampling × rule_gen × refine` grid (2 × 2 × 3 = 12 combos) was run end-to-end.
`llm_coarse` first *appeared* to fail — every question hit the **120s request timeout** — but it
was **not over the token/context limit**; the huge 20-doc-sample prompt just needed more than 120s
on finance's ~88k-token docs. Raising the gpt54 client timeout **120 → 600s** let all 6 llm_coarse
combos complete. Run on the **multi_cluster 86-doc subset** (14 of the 100 docs — the 10-Q/8-K
filings — lack reconstructed JSON), 12 questions; `random` = 18 sampled / 68 unsampled, `fps` =
20 / 66 (fps re-sampled on the 86-doc pool). Cost = combined apply `retrieved/doc`.

| Strategy (sampling / rule_gen / refine) | sAcc | uAcc | combined acc | cost ratio |
|---|---:|---:|---:|---:|
| random / llm_coarse / agentic_codex | 0.880 | 0.869 | **0.871** | 0.0547 |
| random / llm_coarse / p_hybrid | 0.870 | 0.860 | 0.862 | 0.0285 |
| random / llm_coarse / p_mini | 0.875 | 0.854 | 0.859 | 0.1000 |
| fps / llm_coarse / agentic_codex | 0.858 | 0.856 | 0.857 | 0.0213 |
| fps / llm_coarse / p_hybrid | 0.863 | 0.850 | 0.853 | 0.0242 |
| fps / llm_coarse / p_mini | 0.867 | 0.845 | 0.850 | 0.0740 |
| fps / agent_codex / agentic_codex | 0.871 | 0.806 | 0.821 | **0.0056** |
| fps / agent_codex / p_mini | 0.863 | 0.806 | 0.819 | 0.0067 |
| fps / agent_codex / p_hybrid | 0.858 | 0.802 | 0.815 | 0.0067 |
| random / agent_codex / agentic_codex | 0.875 | 0.798 | 0.814 | 0.0059 |
| random / agent_codex / p_mini | 0.861 | 0.797 | 0.810 | 0.0060 |
| random / agent_codex / p_hybrid | 0.866 | 0.794 | 0.809 | 0.0060 |
| fps / **agent_codex_val** / merge ◊ | 0.858 | 0.866 | **0.864** | 0.0061 |

◊ **Validation-guarded** Codex rule-gen (2026-06-14): trains on the 20 sampled docs, then
validates on a held-out 20-doc set (carved from the unsampled pool) and broadens overfit rules
(≤ 3 passes); **no refine**, **`merge`** apply (not part of the 2×2×3 grid). Reported four ways —
sampled 0.858 / **val 0.883** / clean-test (46 docs, no leakage) **0.859** / full-unsampled
**0.866**. It lifts `agent_codex` from **0.806 → 0.866 unsampled (+6 pt)**, essentially **matching
the best `llm_coarse`** (0.871) while keeping agent_codex's cheap inference (0.0061) — but at a
**~$94 rule-gen cost** (70.3M codex gen + 0.18M LLM-verification tokens; ~235× the baseline
`agent_codex`'s ~$0.4, far above llm_coarse), so it is **not Pareto-optimal**. Unlike nopv (where
the guard stayed ~3 pt below llm_coarse), on finance it closes the gap. The 2 hard questions still
drag it: Q6 material-agreement **0.258**, Q8 long-term-debt **0.530**; the other 10 average ~0.96.
Result files: `results/financebench/grid/{apply,rule_gen}/fps/agent_codex_val_gpt54/...`.

**The familiar accuracy↔cost split (as on nopv/court):**
- **`llm_coarse` wins accuracy** — **0.850–0.871** (it ingests the whole 20-doc sample) — but is
  **3–18× pricier** (cost 0.021–0.100; `p_mini` is the most expensive refiner at 0.07–0.10).
- **`agent_codex` wins cost** — **~0.006** (reads docs selectively) — at **0.809–0.821** accuracy,
  ~4–6 pts lower.
- Best accuracy overall: `random/llm_coarse/agentic_codex` (**0.871** @ 0.055). Best value:
  `fps/agent_codex/agentic_codex` (0.821 @ **0.0056**) or `fps/llm_coarse/agentic_codex`
  (0.857 @ 0.021). `agentic_codex` is the strongest+cheapest refiner in **both** families.
- Both grid families still trail `agentic_full_data_adaptive` (**0.923**) on accuracy — its agentic
  large-sample rule-gen beats the fixed 20-doc grid sample on finance — but every grid combo is far
  cheaper than the per-pair baseline (1.45).
- **Key fix:** llm_coarse's prompt fits context fine; the failure was a 120s *timeout*, not a token
  limit — the 120→600s client-timeout change made it viable (it ran slow but clean, 0 timeouts).

#### Same 12 combos on the 10 easy questions

Dropping the 2 structurally hard questions (long-term debt, exhibit/material-agreement) — the same
easy-10 subset used above — recomputed for all 12 grid combos:

| Strategy (sampling / rule_gen / refine) | acc (10 easy) | cost ratio |
|---|---:|---:|
| random / llm_coarse / agentic_codex | **0.959** | 0.0225 |
| random / llm_coarse / p_hybrid | 0.953 | 0.0114 |
| random / llm_coarse / p_mini | 0.951 | 0.0664 |
| fps / llm_coarse / p_hybrid | 0.949 | 0.0117 |
| fps / llm_coarse / agentic_codex | 0.947 | 0.0146 |
| fps / llm_coarse / p_mini | 0.943 | 0.0640 |
| fps / agent_codex / p_mini | 0.917 | 0.0030 |
| fps / agent_codex / agentic_codex | 0.916 | **0.0021** |
| fps / agent_codex / p_hybrid | 0.915 | 0.0030 |
| random / agent_codex / p_hybrid | 0.907 | 0.0018 |
| random / agent_codex / agentic_codex | 0.906 | **0.0017** |
| random / agent_codex / p_mini | 0.906 | 0.0018 |

**On the easy 10, every grid combo clears 0.90.** llm_coarse rises to **0.943–0.959** (≈ +0.09 vs the
all-12 numbers) and agent_codex to **0.906–0.917** (≈ +0.10) — confirming the 2 hard questions were
the main drag on both families. The tradeoff persists: llm_coarse ~0.95 at 0.011–0.066, agent_codex
~0.91 at **~0.002** (≈5–30× cheaper). Best easy-10: `random/llm_coarse/agentic_codex` (0.959 @ 0.0225);
best value: `random/agent_codex/agentic_codex` (0.906 @ **0.0017**). `p_hybrid` is the cheapest
llm_coarse refiner here (0.949–0.953 @ ~0.011).

### FINANCEBENCH — ALL strategies on the 10 easy questions (unified)

Every financebench strategy on the same easy-10 subset (12 multi_cluster questions minus
long-term debt + exhibit/material-agreement), sorted by accuracy. **Doc scope differs by group**
(baselines: ~50/49 single-cluster docs; `agentic_full_data`: all 100; legacy raw + grid: 18/68
random or 20/66 fps) — so accuracy is comparable as a *level*, but cost ratios reflect each run's
scope. Cost is the per-pair apply ratio (baselines read whole docs → ≫ 1).

| Strategy | Model / refine | Accuracy | Cost ratio |
|---|---|---:|---:|
| Baseline 1 — Codex QA (per-pair) | gpt54 | **0.986** | 1.45 |
| Baseline 1 — Codex QA (per-pair) | gpt54mini | 0.976 | 1.30 |
| all_docs / agentic_full_data_adaptive | gpt54 | 0.967 | 0.0135 |
| Baseline 2 — Codex QA All | gpt54mini | 0.960 | 0.17 |
| random / llm_coarse / agentic_codex | gpt54 | 0.959 | 0.0225 |
| Baseline 2 — Codex QA All | gpt54 | 0.957 | 0.15 |
| random / llm_coarse / p_hybrid | gpt54 | 0.953 | 0.0114 |
| random / llm_coarse / p_mini | gpt54 | 0.951 | 0.0664 |
| fps / llm_coarse / p_hybrid | gpt54 | 0.949 | 0.0117 |
| fps / llm_coarse / agentic_codex | gpt54 | 0.947 | 0.0146 |
| multi_clusters / llm_coarse / raw | gpt54 | 0.943 | 0.0540 |
| fps / llm_coarse / p_mini | gpt54 | 0.943 | 0.0640 |
| multi_clusters / llm_coarse / agentic+fallback | gpt54 | 0.940 | **0.0090** |
| all_docs / agentic_full_data | gpt54mini | 0.935 | 0.0141 |
| all_docs / agentic_full_data_adaptive | gpt54mini | 0.923 | 0.0140 |
| fps / agent_codex / p_mini | gpt54 | 0.917 | 0.0030 |
| fps / agent_codex / agentic_codex | gpt54 | 0.916 | 0.0021 |
| fps / agent_codex / p_hybrid | gpt54 | 0.915 | 0.0030 |
| random / agent_codex / p_hybrid | gpt54 | 0.907 | 0.0018 |
| random / agent_codex / agentic_codex | gpt54 | 0.906 | **0.0017** |
| random / agent_codex / p_mini | gpt54 | 0.906 | 0.0018 |

**Takeaway:** on the easy 10 everything is tightly bunched at **0.91–0.99**, so cost is the
differentiator. The per-pair Codex baseline tops accuracy (0.986) but at **1.45** per pair;
`llm_coarse` pipelines reach **0.94–0.96 at 0.01–0.07** (≈20–145× cheaper), and `agent_codex`
holds **0.91 at ~0.002** (≈700× cheaper than the baseline). `agentic_full_data_adaptive` gpt54
(0.967) is the best non-baseline accuracy.

---

## OFFICEQA

> ⚠️ **7 pipelines exist: the 6 `agent_codex` combos + `fps/llm_coarse/p_hybrid`.** The
> `llm_coarse` generator was originally stopped on officeqa (its broad JSON prompt risks
> overflowing the context window on these very large docs), but `fps/llm_coarse/p_hybrid` ran
> successfully — rule-gen reads only the JSON `texts[:80]` slice (~16 docs' worth, RL 16.24) and
> the apply/merge stage truncates retrieved text to ~250k tokens rather than failing. The other 5
> `llm_coarse` combos (`random` + the two other refiners) are not yet run.

officeqa avg doc size ≈ **348,961 tok** (plain text — very large office docs) / **27,816 tok**
(JSON `texts[:80]`, inflation **0.1×** — the first 80 spans are a tiny slice of these huge docs),
16 queries, 697-doc corpus, split **n = 20 sampled / m = 180 unsampled / 200 total**.
Baselines: 16q × 50 docs (n = 800). Column definitions identical to the NOPV section.

| Strategy (sampling / rule_gen / refine) | Model | RL cost ratio | QA accuracy | QA cost ratio |
|---|---|---:|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | N/A | **0.830** | 160.04 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | N/A | 0.796 | 164.41 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | N/A | 0.779 | 20.86 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | N/A | 0.556 | 9.87 |
| fps / llm_coarse / p_hybrid | gpt54 | 16.24 | **0.644** | 0.186 |
| random / agent_codex / p_mini | gpt54 | 0.52 | 0.585 | 0.0054 |
| random / agent_codex / p_hybrid | gpt54 | 0.52 | 0.584 | 0.0054 |
| fps / agent_codex / p_hybrid | gpt54 | 0.06 | 0.572 | 0.0036 |
| fps / agent_codex / p_mini | gpt54 | 0.06 | 0.572 | 0.0036 |
| fps / agent_codex / agentic_codex | gpt54 | 0.06 | 0.569 | 0.0035 |
| random / agent_codex / agentic_codex | gpt54 | 0.52 | 0.555 | 0.0042 |
| all_docs / agentic_full_data_adaptive / merge ◊ | gpt54 | 18.26 | 0.522 | 0.129 |
| all_docs / agentic_full_data_adaptive / merge ◊ | gpt54mini | 15.18 | 0.386 | 0.129 |
| all_docs / agentic_full_data / merge ◊ | gpt54mini | 7.49 | 0.374 | 0.131 |

◊ **Rule End-to-End** (full-corpus agentic generation; see `docs/approach/rule_end_to_end.md`).
A Codex agent generates rules per question by reading the raw `.txt` corpus on demand (no
20-doc sample split), then `rule_apply_merge` answers + judges with **gpt54 throughout** over
**all 200 docs**. So QA accuracy here is the all-docs accuracy (not an `n=20`/`m=180` combined
mean), directly comparable to the grid rows' combined accuracy and the baselines. RL cost ratio =
mean per-query rule-generation `input_tokens` ÷ plain-text avg doc size (348,961 tok). `_adaptive`
= `--adaptive-large-sample` (larger working sample during generation).

*sAcc/uAcc breakdown (pipelines):*

| sampling / rule_gen / refine | sAcc | uAcc | combined acc |
|---|---:|---:|---:|
| fps / llm_coarse / p_hybrid | 0.409 | 0.670 | **0.644** |
| random / agent_codex / p_mini | 0.644 | 0.579 | 0.585 |
| random / agent_codex / p_hybrid | 0.641 | 0.578 | 0.584 |
| fps / agent_codex / p_hybrid | 0.444 | 0.586 | 0.572 |
| fps / agent_codex / p_mini | 0.441 | 0.586 | 0.572 |
| fps / agent_codex / agentic_codex | 0.441 | 0.583 | 0.569 |
| random / agent_codex / agentic_codex | 0.600 | 0.550 | 0.555 |

### Analysis

**OfficeQA is the hardest dataset for the pipeline.** The best pipeline is now
`fps/llm_coarse/p_hybrid` at combined **0.644** (uAcc **0.670**) — still below the best baseline
(Codex per-pair gpt54, **0.830**), a ~19-pt gap, but a clear step up from the `agent_codex`
combos, which top out at **0.585** (`random/agent_codex/p_mini`). `llm_coarse` generalizes
notably better to the held-out set (uAcc 0.670 vs `agent_codex`'s 0.55–0.59) — its broader,
JSON-derived rules cover these heterogeneous office docs more completely — but it pays for that
with a far heavier QA footprint.

**`llm_coarse` buys accuracy with cost.** Its QA cost ratio is **0.186** — roughly **35–50×**
the `agent_codex` pipelines (0.0035–0.0054), because its rules retrieve much more text per doc
(and the apply/merge stage truncates to ~250k tokens on the largest treasury bulletins). Even so
it stays well under the baselines: ~**860×** cheaper than Codex per-pair (160) and ~**112×**
cheaper than *All* gpt54 (20.86). So `llm_coarse` is the accuracy-leaning point on officeqa's
Pareto front, `agent_codex` the cost-leaning one.

**But the cost asymmetry is extreme** (for the `agent_codex` family). Those pipelines answer at
cost ratio **0.0035–0.0054** — because they retrieve a sliver of a ~349K-token doc — while the
baselines pay **160–164** (per-pair) or **10–21** (amortized *All*). So `agent_codex` is
**~30,000–45,000× cheaper** than the per-pair baseline and **~2,000–4,000× cheaper** than the
*All* baseline, but at ~0.58 vs 0.78–0.83 accuracy. Notably, even the `agent_codex` pipelines
(0.555–0.585) **beat Baseline 2 *All* gpt54mini** (0.556) outright, and `fps/llm_coarse/p_hybrid`
(0.644) clears it comfortably — approaching the *All* gpt54 baseline (0.779) on accuracy at a
fraction of its cost.

**Sampling/refine effects:** `random` sampling gives higher combined accuracy for the Pareto
refiners (0.584–0.585 vs fps 0.569–0.572) but costs ~10× more to learn (RL 0.52 vs 0.06 —
fps Codex read almost nothing). `agentic_codex` refine is the weakest accuracy on both samplers.

**Rule End-to-End (full-corpus agentic generation) underperforms the grid on officeqa.** Reading
the *entire* corpus during generation — rather than a 20-doc sample — does not pay off here: the
best variant, `agentic_full_data_adaptive` (gpt54), reaches only **0.522**, below both
`fps/llm_coarse/p_hybrid` (**0.644**) and every `agent_codex` grid combo (0.555–0.585), while
costing far more on both axes (RL **7.5–18.3** vs ≤0.52; QA **0.13** vs 0.0035–0.0054). The
`gpt54mini` generators are weaker still (0.374–0.386). officeqa's docs are large and
heterogeneous enough that more reading during generation mostly adds cost, not transferable rule
quality — the opposite of FinanceBench, where the same full-data adaptive approach tops the
pipelines (0.923). The generation model matters far more than sample size: gpt54 over gpt54mini
is +13.6 pts (0.386→0.522), the adaptive sample only +1.2 pts on gpt54mini.

### OFFICEQA — normalized (single cost ratio)

Computed exactly as in the NOPV normalized section: `Accuracy = (20·sAcc + 180·uAcc)/200`;
`Cost ratio = (20·(RL/20) + 180·unsampled_cr)/200`.

| Strategy (sampling / rule_gen / refine) | Model | Accuracy | Cost ratio |
|---|---|---:|---:|
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54 | **0.830** | 160.04 |
| **Baseline 1** — Agentic Codex QA (per-pair) | gpt54mini | 0.796 | 164.41 |
| **Baseline 2** — Agentic Codex QA All | gpt54 | 0.779 | 20.86 |
| **Baseline 2** — Agentic Codex QA All | gpt54mini | 0.556 | 9.87 |
| fps / llm_coarse / p_hybrid | gpt54 | **0.644** | 0.248 |
| random / agent_codex / p_mini | gpt54 | 0.585 | 0.0074 |
| random / agent_codex / p_hybrid | gpt54 | 0.584 | 0.0074 |
| fps / agent_codex / p_hybrid | gpt54 | 0.572 | 0.0036 |
| fps / agent_codex / p_mini | gpt54 | 0.572 | 0.0036 |
| fps / agent_codex / agentic_codex | gpt54 | 0.569 | 0.0034 |
| random / agent_codex / agentic_codex | gpt54 | 0.555 | 0.0063 |
| all_docs / agentic_full_data_adaptive / merge ◊ | gpt54 | 0.522 | 0.220 |
| all_docs / agentic_full_data_adaptive / merge ◊ | gpt54mini | 0.386 | 0.204 |
| all_docs / agentic_full_data / merge ◊ | gpt54mini | 0.374 | 0.168 |

◊ Rule End-to-End rows use a different amortization base: full-corpus generation has no 20-doc
sample, so `Cost ratio = RL/200 + unsampled_cr` (one-time per-query generation amortized over the
200-doc corpus, plus per-pair apply cost). Accuracy is the all-docs value. See the ◊ footnote on
the unnormalized table above and `docs/approach/rule_end_to_end.md`.
