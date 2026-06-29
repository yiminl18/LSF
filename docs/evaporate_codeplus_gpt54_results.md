# Evaporate Baseline — Code+ / gpt-5.4 Results

**Variant:** `codeplus` (top-k synthesized functions + Snorkel weak-supervision combine)
**Model:** `gpt-5.4` via Pioneer (`api.pioneer.ai`, OpenAI-compatible), same model for synthesis/extraction and judge
**Sampling:** random, seed=0, cap=20 sampled docs (FinanceBench uses the prebuilt `multi_cluster` split)
**Run date:** 2026-06-26 · **Box:** Singapore GPU box (eval is API-bound, GPU unused)

## Results (all costs measured, not estimated)

| dataset | gen cost | judge cost | total cost | sampled acc | unsampled acc | total acc |
|---|--:|--:|--:|--:|--:|--:|
| financebench | $7.26 | $0.57 | **$7.83** | 0.400 | 0.381 | **0.385** |
| nopv | $7.55 | $1.03 | **$8.58** | 0.350 | 0.350 | **0.350** |
| court | $7.37 | $1.28 | **$8.65** | 0.235 | 0.199 | **0.202** |
| officeqa | $9.73 | $0.90 | **$10.63** | 0.050 | 0.043 | **0.044** |
| product | $7.62 | $2.31 | **$9.93** | 0.442 | 0.443 | **0.443** |
| tropic | $8.46 | $0.94 | **$9.39** | 0.307 | 0.307 | **0.307** |
| **total** | **$47.99** | **$7.03** | **$55.01** | | | |

Splits: financebench 12 q · 20 sampled / 80 unsampled (100 docs); nopv 12 q · 20 / 222 (242); court 13 q · 20 / 274 (294); officeqa 16 q · 20 / 180 (200); product 13 q · 20 / 180 (200); tropic 14 q · 20 / 180 (200).
Judge calls: financebench 1200, nopv 2904, court 3822, officeqa 2562, product 2600, tropic 2800 (docs×questions minus (doc,question) pairs with no ground truth, which skip the judge call).

**product / tropic** (added 2026-06-29): these come from the yiming-dev `data/product` (EU drug EPAR product-information docs) and `data/tropic` (NHC tropical-cyclone reports) datasets — 200 labeled docs each. Documents were docling-reconstructed on the GPU box (identical to yiming-dev's `data/<ds>/json`, verified); evaluated with `--processing-dir datasets/{epar,nhc_tcr}/processing`. The split is **not** seed-0 runtime-derived: the 20 sampled docs are the prebuilt list in yiming-dev `data/<ds>/random_sample_20.txt`, materialized into `data/<ds>/sample/single_cluster/random/{sample,unsampled}_doc_labels.json` so `stage_sampling` picks it up.

## Accuracy definitions

- **sampled acc** — accuracy on the 20 sampled docs (synthesis set), averaged across questions.
- **unsampled acc** — accuracy on the held-out docs, averaged across questions.
- **total acc** — per question, pool both splits' docs `(n_correct_sampled + n_correct_unsampled) / (n_sampled + n_unsampled)`, then average across questions.

Judging: gpt-5.4 semantic-equivalence judge (`CORRECT`/`INCORRECT`), temperature 0.

## Cost accounting (pricing $2.50 / $15 per 1M input/output tokens)

- **gen cost** = synthesis + extraction phase, from the per-call token ledger (`be.ledger` → `output.json`). Exact.
- **judge cost** = per-call `resp.usage` captured durably (one JSONL line fsync'd per call). Exact.
- Predictions fed to the judge are capped at 8000 chars (a valid answer is short; only degenerate huge extractions are truncated — they are wrong either way, and the cap prevents 32k-context 400 errors).
- judge ≈ 11% of total; synthesis dominates.

## Method notes

- **Threshold-bypass selection (agreed methodology):** upstream Evaporate keeps only functions with F1 ≥ 0.5; when *no* function clears that bar on a question, we bypass the threshold and force the top-k functions by raw F1, so every question yields evaluable predictions (and Code+ still aggregates). Used on every question for court/nopv and most for financebench (synthesized functions rarely clear 0.5 on these QA-style attributes).
- **Parallelism:** questions run as concurrent `run_variant` subprocesses (`--question-workers 4`); judge calls run concurrently (`--judge-workers 32`). Verified race-free (process isolation for synthesis; order-preserving, stateless map for judging; temperature-0 deterministic verdicts).
- **financebench** is Evaporate's strongest dataset here (total acc 0.385; the document-type question scores 1.000 / 0.975). **officeqa is weakest** (0.044): treasury-bulletin docs are huge (~349K tokens) and the QA-style attributes rarely yield a function clearing F1≥0.5 (threshold-bypass fired on all 16 questions) — consistent with officeqa being the hardest dataset in the LSF grid too. court (0.202) sits between (regex functions rarely match appellate docket formats).
- **officeqa setup:** its raw `data/officeqa/json` is layout JSON (`document.elements`), not the `texts` schema, so it must be converted first: `python scripts/convert_officeqa_to_texts.py --in-dir data/officeqa/json --out-dir data/officeqa/normalized_json`, then run with `--dataset officeqa --processing-dir data/officeqa/normalized_json --queries data/officeqa/queries.json` (matches how the LSF pipeline runs officeqa). officeqa synthesis is ~16 min/question (huge docs) → the full run took ~38 min at `--question-workers 8`.

## Artifacts (on box: `~/LSF/baseline_results/<dataset>/evaporate_codeplus_gpt54/`)

- `pipeline_summary.json` — overall accs, `cost_usd`, `selection_summary`, per-question summary
- `run_metadata.json` — provenance (argv, git, provider, timing, split sources)
- `{question_slug}_{sampled,unsampled}.json` — per-doc predicted / ground_truth / correct / cost_ratio
- `staging/codeplus/.../output.json` — synthesized function source + per-call synthesis ledger + WS stats
- `judge_rejudge_usage.jsonl` — per-call judge token usage (durable)
- `logs/`, `*.console.log` — run logs
