# Cost-Optimal Rule Selection — Implementation Notes

This document maps the cost-optimal rule selection algorithm (see `rule_selection.pdf`) to concrete modules, scripts, files, and on-disk artefacts in the LSF codebase. It bridges the formal write-up and the existing code so the selection algorithm can be implemented as a thin layer over what's already there.

> Paths in this document are written relative to the LSF repo root (`~/Documents/Codebase/LSF/`). The single-cluster, LLM-generated, one-shot variant is used throughout for examples; the agent-generated paths (`rules/financebench_single_cluster/agent/...`) follow the same structure.

---

## 1. Mapping: algorithm symbol &rarr; code

| Symbol | Meaning | Where it lives |
|---|---|---|
| `r` &isin; `R` | A single rule | `rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm/rule_<name>.py`. Each file exports `def rule_<name>(doc: dict) -> list[dict]` (loaded by name in `src/rule_apply_individual.py::_load_rule_fn`). |
| `d` &isin; `D` | A single document | `data/financebench/processing/<DOC_NAME>_reconstructed.json`. Spans live under `doc["texts"]`. Document identity for label lookup is `<DOC_NAME>.pdf`. |
| `T(r, d)` | Text returned by rule `r` on doc `d` | List of span dicts returned by `rule_fn(document)`, sorted by `(page_no, structure.level_index)` and joined as `"\n\n".join(s["text"] for s in spans)`. See `src/rule_apply_individual.py` lines 60-73. |
| `c(r, d)` | Per-(rule, doc) token cost | Raw form: `retrieved_token_count` from `rule_apply_individual` (uses `tiktoken.cl100k_base`). Ratio form (what the codebase actually tracks): `cost_ratio = retrieved_token_count / total_doc_tokens`, recorded in `eval_individual/<slug>/<rule>_eval.json::per_document[*].cost_ratio`. |
| `W_r = Σ_d c(r,d)` | Total cost of a rule | Equivalent to `m · avg_cost_ratio` under the ratio convention. The aggregate is already recorded as `eval_individual/<slug>/<rule>_eval.json::avg_cost_ratio`. Sorting by `W_r` is the same as sorting by `avg_cost_ratio`. |
| `a(r, d)` | Whether rule `r` alone solves doc `d` | `eval_individual/<slug>/<rule>_eval.json::per_document[*].correct` — produced by `src/eval_rule.py` running the LLM-as-judge over `rule_apply_individual` predictions. |
| `cov(r)` | Per-rule coverage = mean of `a(r, ·)` | Top-level `accuracy` in `eval_individual/<slug>/<rule>_eval.json`. |
| `A(S, d)` | Merge-accuracy of subset `S` on doc `d` | Produced by `src/rule_apply_merge.py::rule_apply_merge(document, rule_names, ...)` (applies all rules in `S`, dedupes spans by index in `doc["texts"]`, sorts by `(page_no, level_index)`, runs the QA LLM) followed by the LLM-as-judge step (currently inlined in `test/run_eval_merge_sampled.py::judge(...)`). |
| `A*(d) = A(R, d)` | Merge-accuracy of the full rule set on `d` | `eval_merge/<slug>_10_sampled.json::per_doc[*].correct` when `rule_names` = every rule in the folder. Already populated by `test/run_eval_merge_sampled.py`. |
| `D*` | Docs the full set solves | `{record.doc_name : record.correct == True}` extracted from the same `per_doc` list as above. |
| Ground truth | Reference answer for `(question, doc)` | `data/financebench/sample_doc_labels.json` (and `data/financebench/unsampled_doc_labels.json` for held-out). Schema: `{ "<DOC_NAME>.pdf": { "<question>": "<answer>", ... } }`. |
| Substring proxy for `a(r, d)` | Free stand-in for the judge | Not present in `src/`. Defined only as guidance in `task_prompt_rule_gen.py` (case-insensitive substring of ground truth in retrieved text). The selector can add this as an optional pre-filter to cut LLM-judge calls. |

The cost convention deserves an explicit note. The PDF write-up uses raw token counts; the codebase uses a *ratio* (retrieved / full-doc tokens). Multiplying by a fixed per-doc denominator does not change any ordering or set-cover decision, so the algorithm transfers directly — only the numeric values of `c`, `W_r`, and the upper bound `UB` differ.

---

## 2. What's already implemented (the primitives we reuse)

| Primitive | Where | What it does |
|---|---|---|
| Apply one rule + answer | `src/rule_apply_individual.py::rule_apply_individual(document, rule_name, question_slug, question, model_name="gpt54", rules_dir=..., output_dir=...)` | Loads `rule_<name>` from disk, applies to `document`, sorts spans, runs Azure GPT-5.4 (deployment in `src/models/gpt54.py`) with a financial-QA system prompt, writes one record per call to `{output_dir}/{question_slug}/{rule_name}_individual.json`. |
| Apply a rule set + answer | `src/rule_apply_merge.py::rule_apply_merge(document, rule_names, question_slug, question, ...)` | Loads each rule, applies, **dedupes spans by index in `doc["texts"]`** (falls back to `texts.index(span)`), sorts by `(page_no, level_index)`, joins with `\n\n`, runs the same QA LLM. Returns the prediction plus accounting (`retrieved_token_count`, `latency_seconds`, etc.). |
| Judge per-rule predictions | `src/eval_rule.py::eval_rule(rule_name, doc_names, question, question_slug, ...)` | Reads `_individual.json`, looks up ground truth in `labels_file`, runs an LLM equivalence judge (system prompt in `_JUDGE_SYSTEM`), writes per-rule aggregate `accuracy`, `avg_cost_ratio`, `per_document[*].correct` to `{output_dir}/{question_slug}/{rule_name}_eval.json`. |
| Drive merge eval across questions | `test/run_eval_merge_sampled.py` | For every question in `data/financebench/sample_queries.txt`: loads all rules in the question's folder, applies them in merge mode to every sampled doc, runs the inlined judge, writes `eval_merge/<question_slug>_sampled.json` (with `per_doc[*].correct`) and `eval_merge/summary.json`. The unsampled twin is `run_eval_merge_unsampled.py`. |
| Drive individual eval across questions | `test/run_eval_individual.py` | For every (question, rule) pair: runs `rule_apply_individual` on each sampled doc, then `eval_rule`. End state: every `<rule>_eval.json` has `accuracy` (= cov(r)) and per-doc `correct` flags. |
| Token counting | `src/rule_apply_individual.py::_count_tokens`, `src/rule_apply_merge.py::_count_tokens`, `src/eval_rule.py::_count_tokens` | `tiktoken.get_encoding("cl100k_base")` with a `1.3 *` word-count fallback. |

The headline observation: every value the selection algorithm needs except the cheap-first cover itself is already a function of files written to `results/financebench_single_cluster/llm/gpt54/one_shot/{eval_merge, eval_individual}/`. The selector is a thin orchestration layer over these artefacts plus a few new on-demand calls to `rule_apply_merge`.

---

## 3. Phase-by-phase implementation plan

The variant assumed below is single-cluster, LLM-generated, one-shot. The agent variant just swaps the base path (`rules/financebench_single_cluster/agent/gpt54/{raw,refined}/...` and the matching `results/...` tree).

### Phase 0 — Free preprocessing: rank rules by cost

What to do. For every rule `r` in `rules/.../<slug>_10_llm/`, look up `avg_cost_ratio` from `results/.../eval_individual/<slug>_10_llm/<r>_eval.json` and sort ascending.

If `eval_individual` has not been run for this question, the avg_cost_ratio doesn't exist on disk yet. Two options:

1. **Run `test/run_eval_individual.py` once** — this populates `eval_individual` for every (question, rule) pair. It is expensive because each rule pays the LLM-judge cost, but the resulting `avg_cost_ratio` and `accuracy` are exactly what Phases 0 and 3 need. This is the path that aligns with the existing pipeline.
2. **Compute `avg_cost_ratio` without LLM** — write a helper that does only the rule-apply + tokenisation half of `rule_apply_individual` (skip the QA LLM call, skip the judge). This is genuinely free of LLM calls and is the spirit of "Phase 0 is free" in the write-up.

Suggested new module: `src/cost_profile.py` for option 2:

```python
def compute_rule_costs(
    rules_dir: str,                # e.g. "rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm"
    doc_paths: list[Path],         # docs from data/financebench/processing/*_reconstructed.json
    encoding: str = "cl100k_base",
) -> dict[str, dict]:
    """Return {rule_name: {"W": int, "per_doc": {doc_name: int}, "avg_cost_ratio": float}}.

    Implementation: for each rule, import via importlib (same loader as
    rule_apply_individual._load_rule_fn), call rule(doc), concat span text,
    count with tiktoken. No LLM calls.
    """
```

Persist the output as `results/.../cost_profile/<slug>_10_llm.json` so subsequent phases never recompute it.

### Phase 1 — Baseline: read `A*(d)` from disk

What to do. Read `results/.../eval_merge/<slug>_10_sampled.json` produced by `test/run_eval_merge_sampled.py`. The relevant fields per doc are:

```json
"per_doc": [
  { "doc_name": "AMCOR_2019_10K", "correct": true,  "retrieved_tokens": 1189, ... },
  { "doc_name": "BOEING_2018_10K", "correct": false, ... },
  ...
]
```

Set `D* = {row["doc_name"] for row in per_doc if row["correct"]}`. No new LLM calls — this artefact already exists once `run_eval_merge_sampled.py` has been run with the full rule set.

If you want to refresh `A*` for the *current* rule folder before running the selector, just rerun `run_eval_merge_sampled.py` first; it overwrites the slug's sampled JSON.

Suggested new helper: `src/baseline_targets.py`:

```python
def load_target_docs(eval_merge_path: Path) -> set[str]:
    """Read eval_merge/<slug>_sampled.json and return docs the full rule set solves."""
    data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    return {row["doc_name"] for row in data["per_doc"] if row["correct"]}
```

### Phase 2 — Cost-sorted incremental cover

This is the main new piece. Pseudocode (mirrors PDF &sect;5):

```python
def select_rules_cheap_first(
    rules_sorted: list[str],    # by avg_cost_ratio ascending (Phase 0)
    target_docs: set[str],      # = D* (Phase 1)
    documents: dict[str, dict], # doc_name -> loaded JSON
    question_slug: str,
    question: str,
    rules_dir: str,
    judge_fn,                   # str (gt) x str (pred) -> bool
    labels: dict,
) -> list[str]:
    S: list[str] = []
    U: set[str] = set(target_docs)
    for r in rules_sorted:
        if not U:
            break
        gained = set()
        for d in list(U):
            res = rule_apply_merge(
                document=documents[d],
                rule_names=S + [r],
                question_slug=question_slug,
                question=question,
                rules_dir=rules_dir,
                output_dir="results/.../selector_run/",
            )
            gt = labels.get(d + ".pdf", {}).get(question)
            if judge_fn(question, gt, res["predicted_answer"]):
                gained.add(d)
        if gained:
            S.append(r)
            U -= gained
    return S
```

Two non-obvious points.

First, the inner judge call is the expensive part. Add a substring-match pre-filter (mentioned but not implemented in the codebase) and only escalate to the LLM judge when the ground-truth string appears in `res["retrieved_text"]`. This is the proxy `task_prompt_rule_gen.py` already recommends to the rule-generation agent — applied here at the selector layer.

Second, `rule_apply_merge` already writes its merge predictions to `{output_dir}/{question_slug}/<rule_set_slug>_merge.json`. To avoid polluting the existing `rule_run_merge/` directory, point `output_dir` at a dedicated `results/.../selector_run/` while iterating.

Suggested new module: `src/select_rules_cheap_first.py`. It depends on Phase 0's cost profile, Phase 1's `D*`, `rule_apply_merge`, and a `judge_fn` that can be imported directly from `test/run_eval_merge_sampled.py` (extract the `judge(...)` function into `src/eval_judge.py` for cleanliness).

### Phase 3 — Deferred &tau;-check on the selected set

For each `r &isin; S`, we need `cov(r)`. Two cases:

1. **`eval_individual` has been run** (the common case after `test/run_eval_individual.py`): read `accuracy` directly from `results/.../eval_individual/<slug>_10_llm/<r>_eval.json`. Free.
2. **It hasn't been run**: invoke `rule_apply_individual` + `eval_rule` for each `r &isin; S` only — at most `|S| · m` judge calls instead of `n · m`.

Suggested helper: `src/coverage_check.py`:

```python
def load_or_compute_coverage(
    rule_name: str,
    eval_individual_path: Path,
    fallback: Callable[[], float],   # closure that runs rule_apply_individual + eval_rule
) -> float:
    if eval_individual_path.exists():
        return json.loads(eval_individual_path.read_text())["accuracy"]
    return fallback()
```

If `cov(r) < &tau;` for some `r &isin; S`: ban `r`, resume Phase 2 from the current state. Phase 2 only needs to re-cover the docs that `r` was uniquely responsible for — track per-rule `gained` from above and recompute `U &cap; gained(r)` when banning.

### Optional polish

Two cheap local searches after Phase 3 converges, matching PDF &sect;8:

- **Drop test.** For each `r &isin; S`, call `rule_apply_merge(document=docs[d], rule_names=[x for x in S if x != r], ...)` on the docs `r` was credited with and rerun the judge. If they all stay correct, drop `r`.
- **Swap test.** Try replacing the most expensive `r &isin; S` with the cheapest `r'` not in `S`. Admit the swap iff coverage is preserved on all of `D*` and total `avg_cost_ratio` strictly drops.

A natural home for both: `src/polish_rule_set.py`.

---

## 4. Files to add

```
src/
  cost_profile.py             # Phase 0 (free, no LLM)
  baseline_targets.py         # Phase 1 (reads existing eval_merge artefact)
  eval_judge.py               # Extracted judge(...) from test/run_eval_merge_sampled.py
  select_rules_cheap_first.py # Phase 2
  coverage_check.py           # Phase 3
  polish_rule_set.py          # Optional polish

test/
  run_select_all.py           # Driver: for each question in sample_queries.txt, run the full pipeline

results/financebench_single_cluster/llm/gpt54/one_shot/
  cost_profile/<slug>_10_llm.json     # Cached W_r and per-doc costs
  selector_run/<slug>/...             # Merge prediction outputs from Phase 2 (segregated)
  selected_rules/<slug>_10_llm.json   # Final S + accounting (LLM calls used, UB, τ, etc.)
```

The selected-rules artefact should record at minimum:

```json
{
  "question": "...",
  "question_slug": "...",
  "tau": 0.20,
  "selected_rules": ["rule_..."],
  "selected_avg_cost_ratio_sum": 0.087,
  "covered_docs": ["AMCOR_2019_10K", "..."],
  "uncovered_docs": [],
  "baseline_accuracy": 0.90,
  "selector_accuracy": 0.90,
  "llm_calls": {
    "phase_1_baseline": 10,
    "phase_2_incremental": 27,
    "phase_3_coverage": 35
  }
}
```

---

## 5. End-to-end recipe (per question)

```bash
# Prereqs (existing pipeline):
python test/run_rule_apply_merge_all.py        # produces rule_run_merge/...
python test/run_eval_merge_sampled.py          # produces eval_merge/<slug>_sampled.json  (A*)
# Optional but cheaper-for-Phase-3 if you already have it:
python test/run_eval_individual.py             # produces eval_individual/<slug>/<rule>_eval.json (cov)

# New (this pipeline):
python -m src.cost_profile \
  --rules-dir rules/financebench_single_cluster/llm/gpt54/one_shot/<slug>_10_llm \
  --processing-dir data/financebench/processing \
  --labels-file data/financebench/sample_doc_labels.json \
  --out results/financebench_single_cluster/llm/gpt54/one_shot/cost_profile/<slug>_10_llm.json

python -m src.select_rules_cheap_first \
  --cost-profile results/.../cost_profile/<slug>_10_llm.json \
  --eval-merge   results/.../eval_merge/<slug>_10_sampled.json \
  --eval-individual-dir results/.../eval_individual/<slug>_10_llm/ \
  --rules-dir    rules/.../one_shot/<slug>_10_llm \
  --tau 0.20 \
  --out results/.../selected_rules/<slug>_10_llm.json

# Optional polish
python -m src.polish_rule_set \
  --selected results/.../selected_rules/<slug>_10_llm.json
```

A driver script `test/run_select_all.py` should iterate `data/financebench/sample_queries.txt` and chain these calls, mirroring the structure of `test/run_eval_merge_sampled.py`.

---

## 6. Hyperparameter `&tau;`

`&tau;` is the only knob. Interpretation: minimum acceptable per-rule coverage on the sampled docs. Operationally for `m = 10`:

- `&tau; = 0` disables overfit filtering — the selector returns the cheapest cover regardless of how specialised individual rules are.
- `&tau; = 0.2` is a sensible default (drop rules that fire correctly on fewer than 2 docs — already advised in `task_prompt_rule_gen.py`, "A rule covering fewer than 2 documents should be merged into a broader rule or dropped").
- Above `&tau;_max = min_{d &isin; D*} max_{r : a(r,d)=1} cov(r)`, the problem becomes infeasible. `&tau;_max` can be computed exactly when `eval_individual` is on disk: for each `d &isin; D*` find the max `cov(r)` among rules with `per_document[d].correct == True`, then take the min over `d`.

Sweeping `&tau;` across `[0, &tau;_max]` traces the cost-vs-overfit-robustness Pareto curve — useful for sensitivity analysis on the unsampled split (`results/.../eval_merge/<slug>_10_unsampled.json`).

---

## 7. LLM-call budget summary

Let `m = 10` (sampled docs), `n = |R|` (rule pool size, often 50-100 in this codebase based on the rule folders), `k = |S|` (selected size).

| Phase | LLM calls if `eval_individual` is pre-populated | LLM calls if not |
|---|---|---|
| 0. Cost ranking | 0 (read from disk) | 0 (compute via tiktoken only) |
| 1. Baseline `A*` | 0 (read from disk) | `m` (one merge + judge per doc, via `rule_apply_merge` + judge) |
| 2. Incremental cover | `Σ_i u_i` &le; `n · m`, in practice much less | same |
| 3. Coverage check on `S` | 0 (read from disk) | `k · m · 2` (apply + judge per rule per doc) |
| 4. Optional polish | small | small |

For a typical run with `n &asymp; 100`, `k &asymp; 5`, `m = 10`:
- Naive pipeline that ignores existing artefacts: `n · m + m &asymp; 1010` LLM calls.
- Cheap-first with `eval_merge` cached: `~ Σ_i u_i` calls — usually 30-80 in practice.
- Cheap-first with both `eval_merge` and `eval_individual` cached: only the Phase 2 sweep counts as *new* LLM calls.

The win is dominated by replacing `n` with `k` in the dominant term and harvesting prior pipeline outputs for Phases 0, 1, and 3.

---

## 8. Sanity checks before trusting a run

- Phase 0: assert `sum(W_r for r in rules)` equals the sum of `retrieved_token_count` across all `<rule>_individual.json` records for the question slug (allowing fallback-tokenisation drift).
- Phase 1: confirm `|D*| / m` matches `eval_merge/<slug>_sampled.json::accuracy` for the same slug.
- Phase 2: at termination, assert `D* &subseteq; &cup;_{r∈S} gained(r)` and rerun `rule_apply_merge(rule_names=S)` + judge on all of `D*` to confirm every doc is still correct.
- Phase 3: re-read `eval_individual/<slug>/<r>_eval.json::accuracy` for every `r ∈ S` after ban-and-resume, assert all values &ge; `&tau;`.
- Cross-check: `selector_accuracy` recomputed via a full merge eval on `S` must be &ge; `baseline_accuracy` from `eval_merge/.../summary.json`.

---

## 9. Open implementation notes

- **Judge extraction.** The judge function and its system prompt are currently duplicated between `src/eval_rule.py::_JUDGE_SYSTEM` and `test/run_eval_merge_sampled.py::_JUDGE_SYSTEM` (and the `judge` helper). Pulling them into `src/eval_judge.py` benefits both the selector and the existing pipelines.
- **Substring proxy.** Not implemented in `src/`. Add as an optional `proxy_judge(question, gt, retrieved_text) -> bool` in `src/eval_judge.py` to allow Phase 2 to skip the LLM judge on docs where the ground-truth substring is missing.
- **question_slug skew.** Rule folders use the suffix `_10_llm` (or `_10_agent`); run/eval folders use `_10`. `src/eval_rule.py::_load_predictions` handles this with a strip-and-glob fallback. New modules should follow the same convention: accept the rule-folder slug (with `_llm`/`_agent`) and derive the run slug by stripping the suffix when needed.
- **Multi-cluster mix variant.** The mix-doc pipeline lives under `rules/financebench_multi_clusters/...` and `results/financebench_multi_clusters/...`. The algorithm is identical; only the base paths differ. A single `--variant {single,mix}` flag on the selector entry point is the cleanest abstraction.
