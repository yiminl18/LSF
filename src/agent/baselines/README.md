# Agent Baselines

This directory contains document QA baselines that can be run through
`agent.run_pipeline`. They are Phase B only: they do not discover rules, and they
score each `(query_idx, doc_id)` pair directly.

## Common Runner

Run from the repository root:

```bash
PYTHONPATH=src python3 -m agent.run_pipeline \
  --experiment baseline-qa-agent \
  --queries 0 \
  --max-docs 1
```

Useful common flags:

- `--experiment`: one of `baseline-exit`, `baseline-deepread`, `baseline-mdocagent`, `baseline-qa-agent`
- `--config`: experiment YAML, default `src/agent/config_pdfs_10doc.yaml`
- `--queries`: comma-separated zero-based query indices, for example `0,1,2`
- `--max-docs`: cap documents per query
- `--llm-provider`, `--llm-model`: extractor LLM, default `azure/gpt-5.4-mini`
- `--eval-provider`, `--eval-model`: scorer LLM, defaults to the LLM provider/model
- `--seed`: global random seed, default `42`
- `--baseline-output-root`: default `output/agent/baselines`
- `--dry-run`: print planned docs without LLM calls

Each run writes:

```text
<baseline-output-root>/<dataset>/<baseline-name>/q<query_idx>/baseline_rows.jsonl
<baseline-output-root>/<dataset>/<baseline-name>/q<query_idx>/baseline_summary.json
```

For example, `baseline-qa-agent` writes under
`output/agent/baselines/nopv/qa-agent/q0/` when `dataset: nopv`.

## Dataset Expectations

The config should point at a dataset root with this layout:

```text
datasets/<name>/latest/
  queries.txt            # or queries.json with [{"text": "..."}]
  raw/<doc_id>.pdf       # required by PDF/OCR baselines
  processing/<doc_id>_reconstructed.json
  label/10k_q<idx>_reconstructed_labels.json
```

`parser: mineru` switches `processing` and `label` to `processing_mineru` and
`label_mineru`.

`DocInputs` are built from `processing` when available. If reconstructed JSON is
missing, the loader tries to extract text from `raw/<doc_id>.pdf` with PyMuPDF.
If labels are missing, `ground_truth` is empty and the judge/scorer may fail or
mark the row as incorrect.

## Unlabeled Majority-Vote Eval

For datasets without labels, run the extractors directly and score answers by
cross-baseline consensus:

```bash
PYTHONPATH=src python3 -m agent.baselines.majority_vote_eval \
  --datasets nopv,court \
  --queries-per-dataset 5 \
  --docs-per-query 5 \
  --baselines exit,deepread,mdocagent,qa-agent \
  --seed 42 \
  --llm-provider azure \
  --llm-model gpt-5.4-mini \
  --embed-provider openrouter \
  --embed-model openai/text-embedding-3-small \
  --max-doc-pages 2 \
  --deepread-max-pages 2
```

For deterministic front-matter smoke tests, use explicit query indices:

```bash
PYTHONPATH=src python3 -m agent.baselines.majority_vote_eval \
  --datasets nopv,court \
  --query-indices 0 \
  --docs-per-query 1 \
  --baselines exit,deepread,mdocagent,qa-agent \
  --seed 42 \
  --llm-provider azure \
  --llm-model gpt-5.4-mini \
  --embed-provider openrouter \
  --embed-model openai/text-embedding-3-small \
  --max-doc-pages 2 \
  --deepread-max-pages 2 \
  --include-trace
```

Outputs are written under `output/agent/baselines_majority_vote/<run-name>/`:

```text
sample_plan.json
baseline_rows.jsonl
pair_majority_rows.jsonl
summary.json
```

`majority_vote_correct` is a pseudo-label: a baseline is marked correct only
when its answer belongs to the majority-equivalent answer cluster for the same
`(dataset, query_idx, doc_id)`. By default the runner uses a small LLM
equivalence check so short answers and full-sentence answers can vote together
(for example, `August 29, 2024` and `The notice was issued on August 29, 2024`).
Use `--no-llm-vote` to fall back to exact normalized-string voting. Ties are
left unresolved.

## baseline-exit

Implementation: `exit/`.

EXIT is an extractive-compression baseline:

1. Split document text into sentences.
2. Select relevant sentences with the upstream Gemma checkpoint when available.
3. Fall back to LLM zero-shot sentence relevance classification otherwise.
4. Ask the configured reader LLM to answer from the compressed context.

Run:

```bash
PYTHONPATH=src python3 -m agent.run_pipeline \
  --experiment baseline-exit \
  --queries 0 \
  --max-docs 1
```

Notes:

- Uses `doc_inputs.normalized_text` first.
- Falls back to direct PDF text extraction when normalized text is empty and the PDF exists.
- Upstream checkpoint path is optional; the LLM fallback keeps the baseline runnable.
- Set `LSF_EXIT_REQUIRE_GEMMA=1` for GPU verification runs where EXIT must fail
  instead of falling back when the Gemma checkpoint or upstream code is missing.

## baseline-deepread

Implementation: `deepread/`.

DeepRead is an OCR plus locate/read baseline:

1. Render PDF pages and OCR them with a vision LLM into a paragraph index.
2. Run a small locate/read loop with `Retrieve` and `ReadSection`.
3. Synthesize a final answer from gathered evidence.

Run:

```bash
PYTHONPATH=src python3 -m agent.run_pipeline \
  --experiment baseline-deepread \
  --queries 0 \
  --max-docs 1
```

Useful flags:

```bash
--deepread-max-pages 3
--ocr-provider azure
--ocr-model gpt-5.4-mini
```

Notes:

- Default OCR is `azure/gpt-5.4-mini` through the `AZURE_54MINI_*` environment variables.
- Azure GPT-5.4-family OCR calls use `max_completion_tokens`.
- OCR cache is stored under `.cache/deepread_ocr/`.
- This baseline needs source PDFs for OCR. If OCR cannot produce paragraphs, the extractor falls back to normalized text.

## baseline-mdocagent

Implementation: `mdocagent/`.

MDocAgent wraps the upstream multi-modal multi-agent baseline. It requires the
upstream submodule and its environment.

Setup:

```bash
git submodule update --init src/agent/baselines/mdocagent/upstream
cd src/agent/baselines/mdocagent/upstream/MDocAgent
bash install.sh
cd -
```

Run:

```bash
PYTHONPATH=src python3 -m agent.run_pipeline \
  --experiment baseline-mdocagent \
  --queries 0 \
  --max-docs 1 \
  --llm-provider openrouter \
  --llm-model openai/gpt-4o
```

Notes:

- See `mdocagent/README.md` for upstream-specific setup.
- The adapter prepares the document package consumed by upstream MDocAgent.
- The wrapper generates a runtime Hydra model config so `--llm-model` controls
  the model used by all MDocAgent agents.
- The generated runtime model config points at the LSF OpenAI-compatible adapter,
  which handles OpenRouter and Azure GPT-5.4-family `max_completion_tokens`.
- Upstream output is separate from the baseline summary files.

## baseline-qa-agent

Implementation: `qa_agent/`.

QA Agent is the agentic document QA baseline. The model receives document tools
and answers one question for one document. It does not use the rule runtime.

Current tool surface:

```python
semantic_search(query: str, top_k: int = 5) -> list[dict]
keyword_search(query: str, top_k: int = 5) -> list[dict]
regex_search(pattern: str, top_k: int = 20, case_sensitive: bool = False) -> list[dict]
read_chunk(chunk_id: str) -> str
read_pages(start: int, end: int) -> str
python(code: str, timeout_s: int = 3, max_chars: int = 4000) -> str
```

The search tools share the same stable chunk layer. Search hits return
`chunk_id`, `score`, `preview`, `preview_chars`, `total_chars`,
`truncated_chars`, `is_truncated`, and `page` when page metadata is available.
If a hit is truncated, `truncation_note` says how many characters were omitted.
Search tools are locators; `read_chunk` is the main deep-read path after search.
The runtime blocks a final answer based only on truncated search previews and
asks the agent to call `read_chunk`, `read_pages`, or `python` first. Chunk IDs
use `p{page}_c{idx}` when page metadata exists, otherwise `c{idx}`.

Run:

```bash
PYTHONPATH=src python3 -m agent.run_pipeline \
  --experiment baseline-qa-agent \
  --queries 0 \
  --max-docs 1 \
  --embed-provider openrouter \
  --embed-model openai/text-embedding-3-small
```

Embedding provider/model resolution:

1. CLI `--embed-provider` / `--embed-model`
2. `LSF_QA_AGENT_EMBED_PROVIDER` / `LSF_QA_AGENT_EMBED_MODEL`
3. `LSF_EMBEDDING_PROVIDER` / `LSF_EMBEDDING_MODEL`
4. OpenRouter default

Notes:

- Per-document embedding indexes are cached under `.cache/qa_agent_embeddings/`.
- QA Agent does not do runtime embedding-provider fallback. Specify
  `--embed-provider` and `--embed-model` when you want a non-default backend.
- Cache filenames use a document hash, not the source filename.
- The prompt receives anonymous document metadata, not `doc_id` or filename.
- `python` runs in a restricted subprocess sandbox and can inspect `text` and `entries`.
- This baseline works with reconstructed JSON / markdown-like datasets and does not require raw PDFs.
