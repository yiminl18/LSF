# Evaporate baseline (fair vs LSF llm-rule-gen)

Evaporate (Arora et al., *Language Models Enable Simple Systems for Generating
Structured Views of Heterogeneous Data Lakes*, VLDB 2024;
[HazyResearch/evaporate](https://github.com/HazyResearch/evaporate)) as a
baseline that is **same-doc-set comparable to LSF (llm-rule-gen)** = `src/pipeline.py`.

Plan of record: `.omc/plans/evaporate-fair-baseline.md`.
Datasets in scope: **financebench, court, nopv** (officeqa excluded this round).

## What this baseline does (start here)

Evaporate turns each dataset question into an *attribute* and — for the **Code** /
**Code+** variants — **synthesizes small Python extraction functions** from a few
sample docs, then applies them cheaply (no LLM) across the whole collection. The
three variants trade LLM cost for function reuse:

- **`direct`** — no functions; an LLM reads each doc and extracts the answer
  (one LLM call per doc — expensive, like a per-doc QA baseline).
- **`code`** — synthesize one function per question on the sampled docs, then apply
  it with plain Python to every doc (near-zero apply cost).
- **`codeplus`** — synthesize many functions, keep the top-k, and aggregate their
  per-doc outputs by **weak supervision** (a Snorkel `LabelModel`).

We wire it so its accuracy/cost numbers are **directly comparable to LSF
(llm-rule-gen)** — same documents, same judge, same metric (next section).

**Status:** all three variants are smoke-validated end-to-end on `court` with
gpt54mini extraction; the `--model gpt54` backend/deployment is confirmed live (a full
gpt54 end-to-end run is still pending). Full runs additionally need the **`gpt54` judge deployment**
available — see [Run commands](#run-commands).

## Why this is a fair comparison

Evaporate runs its own synthesis + apply (in an isolated venv), but plugs into
**the same three seams** as `pipeline.py`, so its numbers drop into the existing
sampled/unsampled/overall + cost tables with no new scoring code:

1. **Same split.** The orchestrator calls `pipeline.stage_sampling` (random:
   `seed=0`, `cap=20`; or `fps`), so the `sampled`/`unsampled` doc sets are
   byte-identical to the LSF run. Verified: court split == the existing
   `results/court/grid` LSF split (set equality). Evaporate synthesizes on
   `sampled` only.
2. **Same judge.** Predicted answers are scored with pipeline's generic
   `_JUDGE_SYSTEM` / `_judge`, imported directly, running on **gpt54** (NOT mini).
   This is the same judge LSF(llm-rule-gen) used → judge-consistent by construction.
3. **Same per-doc schema + aggregation.** The orchestrator writes pipeline's
   per-doc record schema and per-split aggregates + `pipeline_summary.json`.

**Headline metric:** `avg_unsampled_accuracy` (leakage-free, same-doc-set).
`sampled` and `overall` are secondary.

**Limitation to caption in the writeup:** this is same-split-comparable to
**LSF(llm-rule-gen)** only. It is NOT same-doc-set comparable to the **agentic**
rule-gen runs (`agentic_rule_full_data.py`), which use a runtime `working_sample`
split — there, only unsampled/generalization-level comparison is valid.

## Variants → config

All variants run synthesis/extraction/GOLD_KEY on a **selectable model** (`--model`,
default **gpt54**, or **gpt54mini**) via `runtime/llm_backend.py` — match it to the LSF
variant being compared (LSF's default rule-gen is gpt54). The judge is always gpt54. Attributes
= the dataset's questions (court/nopv: `data/<dataset>/queries.json`; financebench:
`data/financebench/sample_queries.txt`, mirroring `pipeline.py`'s default — there is
no `queries.json` for financebench). Schema-ID is off.

**Thin interception over upstream.** `runtime/run_variant.py` drives the *real
upstream code* — `filter_file2chunks` (keyword chunk search), `get_functions` (all
generation templates), `evaluate` + `get_topk_scripts_per_field` (noisy-LLM-gold F1
with the 0.5 keep-threshold), `apply_final_ensemble` (sandboxed exec), and
`combine_extractions` (MV / abstention) — and intercepts only four seams (N1–N4; see
[`evaporate_deviations.md`](evaporate_deviations.md)). It does **not** reimplement
synthesis/scoring/selection.

| Variant   | Synthesis (on sampled)                              | Apply (on all docs)                          | Apply cost (`retrieved_tokens`)        |
|-----------|-----------------------------------------------------|----------------------------------------------|----------------------------------------|
| `direct`  | none                                                | upstream `get_model_extractions`: LLM reads each doc's chunks, extracts span | per-doc LLM extraction prompt+completion tokens |
| `code`    | upstream `get_functions` (all templates × keyword-filtered chunks); select **best-1** by upstream `evaluate`+`get_topk` (noisy-gold F1, `keep_thresh=0.5`) | best fn applied (pure Python, no LLM)        | `_count_tokens(returned span)`         |
| `codeplus`| same synthesis; keep **top-k** (`get_topk`)         | top-k fns applied; per-doc combine via upstream `combine_extractions` → **weak supervision (snorkel LabelModel)** or MV | `_count_tokens(returned span)`         |

`codeplus` aggregates the top-k functions' per-doc extractions with the modern
**Snorkel `LabelModel` (snorkel 0.10)** — weak supervision. This is the
**maintained successor** to the Snorkel-MeTaL `LabelModel` the original Evaporate
used (MeTaL 0.5 no longer runs under modern networkx — `Graph.node` was removed —
so reviving it is a dead end); it is a successor library, not line-for-line the
2019 code, but the WS algorithm concept is the same. We mirror upstream's `get_data` construction for **label-space construction**: a
**per-document, rank-based local label space** of fixed cardinality (each doc's
classes = its top-N most common distinct extractions, padded with dummies; abstain
= -1, snorkel's convention — note MeTaL used 1..k with 0=abstain).

⚠️ **Class prior (N3 — fairness boundary):** upstream's `get_data` estimates the
class prior from gold over **all** docs (including the apply set = a leak). We
estimate it from the **sampled docs' gold only** — the held-in labels LSF's rule-gen
also uses — by mapping each sampled doc's gold into its local label space and passing
a smoothed `class_balance` to `LabelModel.fit`; unsampled docs vote but contribute no
prior label (no separate apply needed — the top-k functions are already applied to all
docs). Only the apply-set gold is excluded. **Numerically this is ≈uniform** (the
per-doc label space is `random.shuffle`d, so the gold index is randomized), so N3 is
about a correct fairness boundary, not a big number shift; the chosen prior is logged
per question in `codeplus_ws[q].prior`. See
[`docs/baseline/evaporate_deviations.md`](evaporate_deviations.md) for the full record.

`--combiner mv` (plain majority vote) is the **fallback**, used automatically per
question/doc when WS can't apply (**m<3** — snorkel's `LabelModel` needs ≥3 functions,
which MeTaL did not — <2 docs, <2 observed classes, snorkel error, or per-doc
abstain/tie) — counted and reported in `pipeline_summary.json` `codeplus_ws`, never
silent.

The WS shim is installed as `evaporate.profiler.run_ws`, so upstream's own
`combine_extractions` drives it (and its MV fallbacks); we never import the upstream
metal-based `weak_supervision.run_ws` (stubbed in `run_variant._install_stubs`).
Snorkel-MeTaL / cvxpy / metal and manifest-ml are intentionally not installed, and
all LLM calls go through `llm_backend` (patched onto `evaporate.utils.get_response`),
not Manifest.

**Synthesis context = upstream keyword search.** Function synthesis uses upstream
`filter_file2chunks` (keyword search over the attribute string, ≤2 chunks/file —
the paper's §3.3 context selection). The runner does **not** localize on the gold
substring (an earlier reimplementation did; reverted — see "Not deviations anymore"
in [`evaporate_deviations.md`](evaporate_deviations.md)).

Knobs (orchestrator flags): `--model {gpt54,gpt54mini}` (default gpt54; match the LSF
variant), `--chunk-chars` (3000, upstream default), `--num-functions`
(10), `--topk` (10, upstream default), `--max-extract-chunks` (40, Direct only), `--combiner {ws,mv}`
(default `ws`, Code+ only), `--judge-model` (default `gpt54`).

## Cost accounting (two columns — Gap C)

Reported separately, never blended:

- **Synthesis cost** — gpt54mini function-generation tokens (phase=`synthesis`
  in the ledger), amortized over the sampled set. **Direct = N/A.**
- **Apply cost** — `cost_ratio = retrieved_token_count / _count_tokens(doc)`,
  pipeline's exact denominator (`cl100k_base`).

⚠️ **The apply `cost_ratio` is NOT directly comparable across variants** — the
`retrieved_token_count` numerator has different semantics per variant (plan C2):
Direct counts per-doc LLM *extraction* tokens (a real apply-time LLM call),
while Code/Code+ count the *returned-span* tokens (apply is pure Python, no
apply-time LLM). Read the apply column alongside the synthesis column and the
variant, not as a single ranking number.

Synthesis vs apply are distinguished by the `phase` field that `llm_backend`
tags onto every completion.

## Dependency isolation (Gap B)

- **Source:** git submodule `src/baseline/evaporate/upstream` →
  `HazyResearch/evaporate` @ `59eda5d34415f71ffb3a72bb902e1505d1f59e83`.
  Nested submodule `metal-evap` (SSH, `git@github.com:simran-arora/metal-evap`)
  is **left uninitialized** — not needed (Code+ WS uses snorkel, not MeTaL).
- **Deps:** isolated `.venv-evaporate` + subprocess. `runtime/llm_backend.py`
  reads `local/azure.json` (`key_file_cheap`) standalone; it cannot import the
  main repo's `models.*`.
- The orchestrator (`run_evaporate.py`) runs in the **main** env (it needs
  `pipeline.py` + `models.gpt54` for the judge); only `run_variant.py` runs in
  the venv.

## Files

```
src/baseline/evaporate/
  run_evaporate.py      # orchestrator (main env): split + stage + judge + emit pipeline schema
  upstream/             # git submodule (Evaporate source)
  runtime/
    setup_env.sh        # builds .venv-evaporate, runs the A4 self-test
    llm_backend.py      # standalone gpt54mini client + phase-tagged ledger
    run_variant.py      # in-venv subprocess target (Direct/Code/Code+)
    .venv-evaporate/    # gitignored
```

## Run commands

> **⚠️ The judge needs a `gpt54` deployment.** Scoring runs on **gpt54** (not mini).
> If your `local/azure.json` resource has no `gpt-5.4` deployment, the judge step
> fails with `404 DeploymentNotFound`. For a *plumbing-only* smoke you may pass
> `--judge-model gpt54mini` to exercise the chain — but **mini-judged numbers are not
> fair and must never go in a results table**; real comparisons require the gpt54 judge.

```bash
# 1) one-time: build the isolated venv (also runs the A4 import/exec self-test)
bash src/baseline/evaporate/runtime/setup_env.sh

# 2) offline self-test (no Azure calls): upstream import + N1/N2 patch install + WS shim
src/baseline/evaporate/runtime/.venv-evaporate/bin/python \
  src/baseline/evaporate/runtime/run_variant.py --self-test

# 3) a cheap smoke run (1 question, few unsampled docs) — exercises the live chain
python src/baseline/evaporate/run_evaporate.py \
  --dataset court --variant code \
  --output-dir baseline_results/court/evaporate_code_gpt54mini_smoke \
  --limit-questions 1 --limit-unsampled 3

# 4) full runs (per dataset × variant)
python src/baseline/evaporate/run_evaporate.py --dataset court       --variant direct   --output-dir baseline_results/court/evaporate_direct_gpt54mini
python src/baseline/evaporate/run_evaporate.py --dataset court       --variant code     --output-dir baseline_results/court/evaporate_code_gpt54mini
python src/baseline/evaporate/run_evaporate.py --dataset court       --variant codeplus --output-dir baseline_results/court/evaporate_codeplus_gpt54mini
# (repeat for --dataset nopv)

# financebench — the --cluster Evaporate runs MUST match the cluster of the LSF
# (llm-rule-gen) financebench number that goes into the headline table. Both
# single_cluster and multi_cluster LSF results exist; default is single_cluster.
python src/baseline/evaporate/run_evaporate.py --dataset financebench --cluster single_cluster --variant code --output-dir baseline_results/financebench/evaporate_code_gpt54mini
```

> **Naming gotcha (financebench cluster).** The data split dir is
> `data/financebench/sample/multi_cluster/` — **SINGULAR**. There is no
> `multi_clusters` data dir (the plural appears only as an output label). To run
> multi-cluster you must pass `--cluster multi_cluster` (singular); the plural
> fails to resolve the prebuilt split. The orchestrator asserts
> `data/financebench/sample/<cluster>/random/` exists and raises a clear error
> otherwise (rather than silently building a different split from `all_labels.json`).
> The default `--cluster single_cluster` matches pipeline.py's default.

Output mirrors a pipeline run: `pipeline_summary.json` (per-question +
`overall.avg_unsampled_accuracy` / `avg_sampled_accuracy` / `avg_cost_ratio`,
plus a `cost_columns` block with the synthesis/apply split and the gpt54mini
ledger), and per-split `<question_slug>_<split>.json` with the per-doc schema.

## Acceptance criteria status

| # | Criterion | How | Status |
|---|-----------|-----|--------|
| 1 | Split identity | by construction — reuses `pipeline.stage_sampling` (same split as the LSF run) | ✅ by construction |
| 2 | Same judge | imports `pipeline._JUDGE_SYSTEM`/`_judge` verbatim | ✅ by construction |
| 3 | Schema conformance | emits pipeline's per-doc record schema directly | ✅ coded |
| 4 | Completeness guard | orchestrator asserts every doc in `sampled∪unsampled` has a prediction; writes `evaporate_FAILED.json` on shortfall | ✅ coded |
| 5 | Model-split (synth/extract=mini, judge=gpt54, 0 judge on mini) | ledger `by_phase` + `judge_model=gpt54` | ▶ needs a run |
| 6 | Backend routing (`local/azure.json` only) | ledger `credential_source` | ▶ needs a run |
| 7 | Clean-clone A4 | `run_variant --self-test` in venv | ✅ via setup_env.sh |
| 8 | Cost columns | `cost_columns` in summary | ✅ coded (values need a run) |
| 9 | Code+ aggregation | `functions[q]` has multiple fns; per-doc = snorkel LabelModel (WS) result, MV fallback counted in `codeplus_ws` | ✅ verified (self-test + offline `combine_extractions(ws)` wiring: m=3 → ws_applied; m=2 → clean MV fallback) |

Criteria 5/6/8-values/9-values require live gpt54mini + gpt54 calls; run the
smoke command above to populate them.

## Provenance / notes

- Upstream pinned SHA recorded above and echoed into each run's
  `pipeline_summary.json` (`upstream_sha`).
- The upstream `manifest`/`metal`/`cvxpy`/`snorkel-metal` deps are NOT installed
  (the imports they sit behind are stubbed): LLM calls go through `llm_backend`,
  and Code+ WS uses the modern **snorkel `LabelModel`** instead of MeTaL. MeTaL
  0.5 is a dead end on modern networkx (`Graph.node` removed → crash), so we do
  not try to revive `metal-evap`/`snorkel-metal`.
- **Pre-existing caveat flagged to the team (not fixed here):** existing
  LSF-vs-other-baseline tables mix pipeline's generic judge (LSF) with the
  baseline runners' dataset-specific judges — a latent judge mismatch
  independent of Evaporate.
