# LSF: Light-weight Structure Fusion

1. **Problem 1: Document Structure-Aware Retrieval**
2. **Problem 2: Unsupervised Document Clustering**

## Installation

```bash
pip install -e .
```

### Provider Configuration

The artifact code supports multiple provider backends. The required environment
variables depend on the provider you choose.

#### Embedding Providers for Problem 1

`train_model.py` and `evaluate_model.py` support these `--embed-provider` values:
`openai`, `azure`, `openrouter`.

API-backed embedding providers require:

| Provider | Required environment variables |
|----------|--------------------------------|
| `azure` | `AZURE_EMBEDDING_API_KEY`, `AZURE_EMBEDDING_API_BASE`, `AZURE_API_VERSION` |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |

#### LLM Providers

LLM-backed steps support only `--llm-provider azure|openrouter`, and all of
them require an explicit `--model`.

| Provider | Required environment variables |
|----------|--------------------------------|
| `azure` with `gpt-5.4*` | `AZURE_54_API_KEY`, `AZURE_54_API_BASE`, `AZURE_54_API_VERSION`, `AZURE_54_DEPLOYMENT` |
| `azure` with `gpt-5.4-mini*` | `AZURE_54MINI_API_KEY`, `AZURE_54MINI_API_BASE`, `AZURE_54MINI_API_VERSION`, `AZURE_54MINI_DEPLOYMENT` |
| `openrouter` | `OPENROUTER_API_KEY` |

#### Providers for Problem 2

The canonical P2 input builder supports `--embed-provider` and reads provider-
specific document embedding caches from:

- `datasets/pdfs/latest/embedding/<provider>/document_embedding`
- `datasets/paper/latest/embedding/<provider>/document_embedding`

The retained canonical P2 workflow is validated with `--embed-provider openrouter`.

## Problem 1: Document Structure-Aware Retrieval

### Retained Model Surface

This artifact keeps only the two validated ranking model families:

- `xgb-sem-struc-v5`
- `hnn-sem-struc-v5`

The retained feature surface is `mode=25`, a 52-dimensional fusion of semantic,
lexical, structural, content-aware, and visual signals.

### Workflow

> Steps 2-7 require `--parser docling|mineru`.

1. **Preprocess PDFs**

```bash
python -m core.pipeline.preprocess \
    --dataset pdfs \
    --type docling
```

Input: `datasets/<dataset>/latest/raw/*.pdf`  
Output: `datasets/<dataset>/latest/processing/*_docling.json`

2. **Build Processing JSON**

```bash
python -m core.pipeline.build_processing_json \
    --dataset pdfs \
    --parser docling \
    --llm-provider azure \
    --model gpt-5.4-mini
```

Input: `processing/*_docling.json`  
Output: `processing/*_reconstructed.json`

3. **Generate Embeddings**

```bash
python -m core.pipeline.generate_embeddings \
    --dataset pdfs \
    --parser docling \
    --embed-provider openrouter
```

Input: `processing/*_reconstructed.json`  
Output: `embedding/<provider>/document_embedding/*_reconstructed_embeddings.npz`

4. **Generate Labels**

```bash
python -m core.pipeline.generate_labels \
    --dataset pdfs \
    --parser docling \
    --judge-mode answer_compare \
    --llm-provider azure \
    --model gpt-5.4-mini
```

Input: processing JSON + embeddings  
Output: `label/*_reconstructed_labels.json`

5. **Split Dataset**

```bash
python -m core.pipeline.split_dataset \
    --dataset pdfs \
    --parser docling \
    --experiment default
```

Output: `experiments/<exp>/<dataset>/splits/`

6. **Train Models**

```bash
python -m core.pipeline.train_model \
    --dataset pdfs \
    --parser docling \
    --model_config xgb-sem-struc-v5 hnn-sem-struc-v5 \
    --embed-provider openrouter \
    --seeds 41,42,43 \
    --curriculum \
    --experiment default \
    --workers 4
```

Retained training behavior:

- `--curriculum` remains available
- `--seeds 41,42,43` is the standard multi-seed configuration

7. **Evaluate Models**

```bash
python -m core.pipeline.evaluate_model \
    --dataset pdfs \
    --parser docling \
    --model_config xgb-sem-struc-v5 hnn-sem-struc-v5 \
    --embed-provider openrouter \
    --seeds 41,42,43 \
    --score-agg softmax \
    --softmax-alpha 5.0 \
    --experiment default \
    --workers 4
```

Retained evaluation behavior:

- `softmax` is the default score aggregation method
- `alpha=5.0` is the retained default temperature
- `top2_mean` remains available as the simpler alternative

8. **End-to-End Evaluation**

```bash
python -m core.pipeline.e2e \
    --dataset pdfs \
    --parser docling \
    --model-config xgb-sem-struc-v5 \
    --embed-provider openrouter \
    --llm-provider azure \
    --model gpt-5.4-mini \
    --seeds 41,42,43 \
    --experiment default
```

`rag-v1` uses the existing evaluation pipeline without ML models. Chunk-based
baselines are selected with `--model-config rag-vanilla`, `rag-raptor`,
`rag-graph`, or `rag-hippo`, and require `--ref-results-dir` pointing at a
reference xgb-v5 e2e run. `rag-graph` requires `networkx`; `rag-hippo` requires
`spacy` and `en_core_web_sm`.

## Agent Rule Runtime

Agent code is split into three layers:

- `agent.rules`: RangeRule schemas, parsers, execution, and scoring primitives
- `agent.rule_runtime`: shared data packaging, best-rules artifacts, holdout evaluation, and cascade deploy policy
- `agent.reflection_agent` / `agent.tool_agent`: rule-generation strategies that both use the shared runtime and built-in answer scoring

Prompt files follow the same boundary: reflection/baseline rule-generation
prompts are under `prompts/reflection_agent`, and interactive tool-agent prompts
remain under `prompts/tool_agent`.

## Problem 2: Unsupervised Document Clustering

### Canonical Workflow

This artifact keeps exactly one P2 workflow:

1. start from the full reconstructed document corpus
2. generate canonical runtime inputs from that corpus
3. build `S_tfidf` from heading text
4. build `S_tree` from tree-shape fingerprints
5. fuse `S_sem`, `S_tfidf`, and `S_tree` with fixed weights `0.5 / 0.3 / 0.2`
6. run recursive spectral bisection with silhouette pruning
7. run one corpus-level LLM merge over the resulting clusters
8. report NMI, ARI, 10-Q recall, and PERIODIC precision/recall

### Source Corpus

The canonical P2 flow starts from reconstructed processing JSON files:

- `datasets/pdfs/latest/processing-newer` for the full 365-document pdf corpus
- `datasets/paper/latest/processing` for the 238-document paper corpus

There is no separate `processing-full` directory in this repo.

### Generate Canonical Inputs

The retained pipeline does not read processing JSONs directly. It first consumes
four generated runtime inputs:

- `output/phase0/sec_filing_types.csv`
- `output/phase2/representations/full_representations.pkl`
- `output/phase2/similarity_matrices/S_sem_full.npy`
- `output/phase2/clustering/full_corpus_labels.npz`

If the full local datasets are available, build those inputs before running P2:

```bash
python -m core.cluster.bisection.prepare_inputs \
    --force \
    --embed-provider openrouter
```

The canonical builder reads the local processing JSON corpus and the existing
provider-specific document embedding caches already on disk.

If those datasets are not present in this checkout, prepare the four runtime
inputs in `LSF-dev` and copy them into the artifact workspace.

Generated `output/` files are runtime-only and should not be kept in the artifact repo.

### Run the Pipeline

```bash
python -m core.cluster.bisection.pipeline \
    --llm-provider azure \
    --model gpt-5.4-mini
```

The command runs the retained fused clustering pipeline and writes results to
`output/phase2_clustering/`.

### Generated Files

Primary outputs:

- `canonical_pipeline_summary.json`: primary machine-readable summary of the retained methods
- `canonical_pipeline_summary.csv`: tabular convenience view of the same summary

Supporting diagnostics:

- `fused_pruned_confusion.csv`: full-class confusion matrix before LLM merge
- `fused_pruned_periodic_confusion.csv`: periodic-vs-nonperiodic confusion before LLM merge
- `fused_pruned_assignments.csv`: document-to-cluster assignments before LLM merge
- `llm_merged_confusion.csv`: full-class confusion matrix after LLM merge
- `llm_merged_periodic_confusion.csv`: periodic-vs-nonperiodic confusion after LLM merge
- `llm_merged_assignments.csv`: document-to-cluster assignments after LLM merge

### Package Boundaries

- `core.cluster` is the retained Problem 2 runtime and canonical input-builder surface.


## Project Structure

```text
LSF/
├── README.md
├── pyproject.toml
├── datasets/
│   └── .gitkeep
├── experiments/
│   └── .gitkeep
├── src/agent/
│   ├── rules/                  # Rule schemas, execution, and scoring
│   ├── rule_runtime/           # Shared data, prompt, artifact, holdout, deploy runtime
│   ├── reflection_agent/       # Baseline/reflection rule generation
│   ├── tool_agent/             # Interactive tool-based rule generation
│   └── prompts/
├── src/core/
│   ├── doc/
│   ├── embed/
│   ├── retrieval/
│   ├── llm/
│   ├── ml/
│   ├── pipeline/
│   ├── cluster/                # Retained Problem 2 pipeline and input builder
│   └── utils/
└── test/
    ├── test_p1_workflow.py
    ├── test_p2_workflow.py
```

`datasets/` and `experiments/` are intentionally empty in git. They are placeholders for runtime data and generated artifacts.
