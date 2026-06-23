# Evaporate baseline — deviations from the original implementation

This file records **every deliberate deviation** of our Evaporate baseline
(`src/baseline/evaporate/`) from the original Evaporate (Arora et al., VLDB 2024;
upstream `HazyResearch/evaporate`, vendored as a pinned git submodule under
`src/baseline/evaporate/upstream/`).

## Design: thin interception over upstream

The variant runner (`runtime/run_variant.py`) is a **thin interception layer**: it
drives the *real upstream code* for everything scientifically meaningful — keyword
chunk selection (`filter_file2chunks`), function synthesis over all generation
templates (`get_functions`), noisy-LLM-gold scoring with the 0.5 keep-threshold
(`evaluate` + `get_topk_scripts_per_field`), sandboxed function execution
(`apply_final_ensemble`), and MV/abstention combine (`combine_extractions`) — and
**intercepts only the four seams below**. Upstream source is kept pristine (no edits
to the vendored submodule); the interception is done with two runtime monkeypatches
(`evaporate.utils.get_response`, `evaporate.profiler.run_ws`) plus import-time stubs.

The principle: *anything that is not an absolutely-necessary fairness change (or a
forced dependency swap) must use the upstream code, not a reimplementation.* That is
why the list is only four entries — see "Not deviations anymore" at the bottom for
the things an earlier reimplementation got wrong and this rewrite reverted.

## At a glance

| # | Seam | What we change | Reason class | Affects results? |
|---|------|----------------|--------------|------------------|
| N1 | LLM backend | `manifest`→`llm_backend` (selectable gpt54/gpt54mini, default gpt54); never send `stop`; floor output to 256 | fairness + dep + model/API compat | model **must match** the compared LSF variant |
| N2 | Code+ WS engine | Snorkel-MeTaL `LabelModel` → snorkel 0.10 `LabelModel` | dependency rot (forced) | **yes** — successor lib, not the 2019 code |
| N3 | WS class prior | sampled-docs gold only (not all-docs `Y_dev`) | fairness | minor — removes apply-set leak; ≈uniform in practice |
| N4 | scoring / split / cost | final accuracy, doc split, cost → pipeline.py judge/split/schema | fairness/comparability | **yes** — reports LSF-comparable numbers, not Evaporate F1 |

N2 and N4 change reported numbers; read them in full before citing results. N3 only refines the
WS class prior (≈uniform in practice — see below). The gen/apply cost (the gpt54mini phase-tagged
ledger) is reported separately from, and never mixed with, the gpt54 judge cost (a different model
run in the main process — see N4).

---

## N1. LLM backend: `manifest` → `llm_backend` (Azure gpt54 / gpt54mini)

- **Upstream:** all generation/extraction LLM calls go through `manifest-ml`
  (`evaporate.utils.get_response` → `manifest.run`); MODELS / EXTRACTION_MODELS /
  **GOLD_KEY all default to `gpt-4`** (configs.py) — the strong model is used throughout,
  including the noisy-gold scorer that ranks synthesized functions.
- **Ours:** `evaporate.utils.get_response` is monkeypatched to route every upstream
  LLM call through `runtime/llm_backend.py`, a standalone Azure client reading
  `local/azure.json`, with a phase-tagged usage ledger. `apply_prompt` looks up
  `get_response` as a module global at call time, so this one patch reaches every
  upstream call site. The `manifest` import is neutralized with a stub.
- **Model is selectable (`--model`, default `gpt54`).** All three Evaporate LLM roles —
  function synthesis, Direct/GOLD_KEY extraction, and the noisy-gold scorer — go through
  the **same single backend**: `gpt54` (azure.json's main `deployment`) or `gpt54mini`
  (`key_file_cheap`). It **must match the LSF variant** the baseline is compared against
  (LSF's default rule-gen is gpt54, so the default here is gpt54); the judge is always
  gpt54 (a separate model/process — see N4).
- **GOLD_KEY uses the same selected model, not upstream's gpt-4.** When matching
  LSF-gpt54mini (`--model gpt54mini`), the noisy-gold scorer is also gpt54mini — weaker
  than upstream's gpt-4, so function selection (which function survives the 0.5 keep
  filter) is weaker, which can lower accuracy. This is intrinsic to running the baseline
  at the comparison's model budget, not a separate bug; disclosed here.
- Two sub-points folded in here (same model/API-compat class):
  - **`stop` dropped.** gpt-5.4(-mini) rejects `stop` (`400 Unsupported parameter`).
    We never send it; the completion is post-truncated at the stop sequence locally
    (same effect). `src/models/gpt54*.py` also never send `stop`.
  - **Output-budget floor (256).** gpt-5.4(-mini) is a *reasoning* model: upstream's tiny
    `max_toks` (10/100) get consumed by hidden reasoning and return empty. We floor
    `max_completion_tokens` to 256 so the same prompts return text. Function
    generation (`max_toks=500`) is unaffected.
  - **`gold_choices` unsupported.** Upstream `get_response` has a constrained/log-prob
    `gold_choices` branch; the ClosedIE flow (do_end_to_end=False) never uses it, so the
    patch asserts `gold_choices is None` (fail-loud) rather than silently degrading it.
- **Why:** fairness — every variant must use the *same* gpt54mini backend as the rest
  of the comparison; manifest is also a dead dep we keep out of the isolated venv.
- **Impact:** none on the algorithm; this is the crux of backend-fair comparison.

## N2. Code+ weak-supervision engine: Snorkel-MeTaL → snorkel 0.10 `LabelModel`

- **Upstream:** `combine_extractions(combiner_mode="ws")` calls
  `weak_supervision/run_ws.py:run_ws`, which aggregates with **Snorkel-MeTaL**
  (`from metal.label_model import LabelModel`).
- **Ours:** a shim installed as `evaporate.profiler.run_ws` (so upstream's own
  `combine_extractions` drives it, including its MV fallbacks) uses the modern
  **`snorkel.labeling.model.LabelModel` (snorkel 0.10)** — the maintained successor to
  the same Snorkel label-model method. It faithfully mirrors upstream `get_data`'s
  label-space construction: a per-document, rank-based local label space of fixed
  cardinality (`num_elts=5`), top-N most common distinct extractions, dummy padding,
  `random.seed(0)` shuffle, `-1` abstain. Two engine properties:
  - snorkel's `LabelModel` requires **≥3 labeling functions** (MeTaL did not); below
    that, or `<2` docs, the shim cleanly returns abstain and `combine_extractions`
    falls back to majority vote. Default `topk=5` clears this.
  - the cvxpy **structure-learning / dependency** step is dropped — this is upstream's
    own `try/except` "Not modeling dependencies" fallback path (run_ws.py:256-257),
    not a new deviation.
  - **cardinality is fixed at `num_elts=5`**, whereas upstream MeTaL used a dynamic
    `k=len(classes)` derived from the gold class count (run_ws.py:193,200). The label
    space is per-doc and local, so there is no global gold class count to derive `k`
    from; we use the fixed choice count and the "<2 observed classes" guard prevents
    degenerate fits.
- **Why:** dependency rot — **Snorkel-MeTaL 0.5 (2019) no longer runs**: it uses the
  networkx `Graph.node` API removed in networkx ≥2.4 (our venv has nx 3.6), crashing
  `LabelModel.train_model`. The deps install on py3.14 but the engine is broken.
- **Impact:** **not line-for-line the 2019 MeTaL code.** Same WS *concept* (a generative
  label model over a per-doc rank-based label matrix), faithfully reproducing upstream's
  label-matrix construction (verified vs `run_ws.py:get_data`). Label encoding follows
  snorkel's convention (`0..k-1`, `-1` abstain) vs MeTaL's (`1..k`, `0`). Disclosed as a
  successor library, not the original.

## N3. WS class-balance prior: all-docs gold `Y_dev` → SAMPLED-docs gold only

- **Upstream:** `get_data` builds a per-doc **gold** vector (`test_gold`) for **every** doc
  in the apply set from the gold file and passes it as `Y_dev`, so the label model estimates
  the class prior from gold over **all** docs (`run_ws.py:76-87, 207-214`).
- **Ours:** the N2 shim estimates the prior from the **sampled docs' gold only** — the
  held-in labels LSF's rule-gen also uses. For each sampled doc it maps the cleaned gold
  answer to that doc's local label index (exactly upstream `get_data:76-87`, restricted to
  sampled docs; gold not in the doc's local space → a random class, as upstream does), builds
  a `class_balance` vector (add-1 smoothed → no zero-prob class), and passes it to
  `LabelModel.fit`. Unsampled docs vote into `L` but contribute **no** class-balance label.
  When no sampled gold is available it falls back to a uniform `1/k` prior. The gold file is
  never opened — sampled gold arrives in-process via `_SAMPLED_GOLD_BY_ATTR`.
- **No separate apply needed.** The sampled docs already carry function votes in `L`: the
  top-k selected functions are applied to **all** docs (sampled ∪ unsampled) by
  `apply_final_ensemble`, so each sampled doc's gold maps into the same local label space its
  votes built. (If the functions did not surface the gold value on a sampled doc, that doc's
  gold falls back to a random index — upstream `get_data:86`.)
- **Why — the fairness boundary:** the leak is specifically the **unsampled (apply-set)**
  gold; the **sampled** docs are the labeled held-in split both systems may use, so using their
  gold for the prior is legitimate parity (not a leak) and is more faithful to upstream than
  dropping gold entirely. Only the apply-set gold is excluded.
- **Impact — minor, ≈uniform in practice.** Upstream's label space is per-doc and
  `random.shuffle`d, so the gold's class *index* is randomized; the gold-derived prior (whether
  sampled-only or all-docs) is itself ≈ uniform. This deviation is about a **correct fairness
  boundary**, not a large numerical shift. The chosen prior is recorded per question in
  `codeplus_ws[q].prior` (`"sampled_gold"` or `"uniform"`).

## N4. Scoring harness: Evaporate native F1 → LSF pipeline split/judge/schema

- **Upstream:** scores final extractions with its own F1 over the data lake.
- **Ours:** the orchestrator (`run_evaporate.py`) reuses **`src/pipeline.py`'s** three
  seams so the baseline is comparable to LSF (llm-rule-gen): the same split
  (`stage_sampling` → `_build_random_split`, seed=0, cap=20; financebench reads the
  prebuilt cluster split), the same generic gpt54 judge (`pipeline._JUDGE_SYSTEM`/`_judge`),
  and the same per-doc schema + sampled/unsampled/overall aggregation + two-column cost
  (`cost_ratio = retrieved_token_count / _count_tokens(doc)`). Synthesis runs on the
  **sampled** docs only.
- **Sample size is the dominant cost lever (fairness-driven).** The split uses pipeline's
  `cap=20`, vs upstream Evaporate's default `train_size=10` (configs.py). Because synthesis
  + GOLD_KEY noisy-gold extraction run per sampled doc, this makes the **synthesis-phase LLM
  call count ≈ 2× the upstream-default run**. This is a deliberate consequence of "same split
  as LSF" (seam 1), not an oversight — but it is the single biggest driver of reported gen cost,
  so cite gen cost relative to this sample size, not to an upstream-default-`train_size` run.
- **Gen cost vs judge cost are cleanly separable.** The reported gen/apply cost is the
  gpt54mini phase-tagged ledger (`ledger_summary.by_phase.{synthesis,extraction}`), produced
  inside the `.venv-evaporate` subprocess. The judge runs **after**, in the main process, on a
  **different model** (gpt54) via `pipeline._judge` — its calls never enter that ledger, so the
  gen number carries zero judge tokens. (Judge cost is evaluation overhead identical to LSF's
  and is not itself part of the method cost.)
- **Not to be confused with function selection.** Upstream's `evaluate` /
  `get_topk_scripts_per_field` (noisy-LLM-gold F1 + 0.5 keep-threshold) decides *which
  synthesized function to apply*. That is internal Evaporate behavior with no true-gold
  leak, so it is **reused unchanged** — it is not part of N4. N4 is only the *final answer*
  accuracy/split/cost.
- **Why:** fairness/comparability — both systems scored on the same docs, by the same judge,
  with the same cost metric.
- **Impact:** we report LSF-comparable accuracy + cost, **not** Evaporate's native F1.

---

## Not deviations anymore (reverted to upstream in the thin-interception rewrite)

An earlier reimplementation of the runner deviated in places it did not need to. The
thin-interception rewrite reverted each to upstream code, so these are **no longer
deviations**:

| Earlier deviation | Now (upstream code used) |
|---|---|
| synthesis context = chunk containing the gold substring (our invention; a synthesis-time gold advantage) | `filter_file2chunks` — upstream's keyword search over the attribute, ≤2 chunks/file (the paper's §3.3 context selection) |
| single generation template (`METADATA_GENERATION_FOR_FIELDS[-1]`) | `get_functions` iterates **all** templates (`fn_generation_prompt_num=-1` = "all") |
| function scoring vs **true** sampled gold + `score>0` selection | `evaluate` (noisy-LLM-gold + `pick_a_gold_label`) + `get_topk_scripts_per_field` (`keep_thresh=0.5`) |
| no abstention regime | upstream `evaluate`/`combine_extractions` carry the coverage `e` / threshold `τ` abstention (`use_abstension`, `extraction_fraction_thresh`) |
| hand-rolled MV / label-space aggregation | upstream `combine_extractions` (MV) + the N2 shim only for the snorkel engine |

---

## Dependency summary (isolated `.venv-evaporate`)

| Upstream dep | Status | Replacement / reason |
|---|---|---|
| `manifest-ml` | **stubbed** | LLM calls go through `llm_backend.py` (gpt54mini) — N1 |
| `snorkel-metal` (metal 0.5) | **dropped** | broken on modern networkx; → `snorkel` 0.10 `LabelModel` — N2 |
| `cvxpy` + structure-learning | **dropped** | only used by metal `run_ws`; upstream's own no-deps fallback path |
| `snorkel` 0.10 | **added** | modern WS engine (pin `snorkel==0.10.0` for reproducibility) |
| `openai`, `tiktoken`, `bs4`, `lxml`, `pandas`, `numpy`, `tqdm` | kept | profiler/prompt/sandbox code path |

Upstream submodule is pinned to a specific commit (`.gitmodules`). Venv is Python 3.14;
rebuild with `bash src/baseline/evaporate/runtime/setup_env.sh`.

---

*Keep this file in sync with `docs/baseline/evaporate.md` and the module docstring in
`runtime/run_variant.py`.*
