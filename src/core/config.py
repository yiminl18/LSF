# -*- coding: utf-8 -*-
"""
Pipeline default configuration (single data source).

All default directory names, experiment/dataset names, etc. are defined here
to avoid hardcoding within core modules.
"""

import logging as _logging

_logging.basicConfig(
    level=_logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)

# Default output directory name (used when docling_tool / lsf_tool do not specify output_dir).
# Note: Pipeline scripts (preprocess, train_model, etc.) use PathManager for path resolution,
# typically under `experiments/{experiment}/{dataset}/results` or `datasets/{dataset}/latest/processing`.
DEFAULT_OUTPUT_DIR = "result"


# Default dataset and experiment names (PathManager / evaluation / RAG, etc.)
DEFAULT_DATASET = "pdfs"
DEFAULT_EXPERIMENT = "default"

# judge_header cache directory (overridable via LSF_JUDGE_CACHE_DIR environment variable)
import os as _os

JUDGE_CACHE_DIR = _os.environ.get("LSF_JUDGE_CACHE_DIR", ".cache")

# === Embedding Models ===
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# === API Pricing (USD per million tokens) ===
EMBEDDING_PRICE_PER_MILLION = 0.02
GPT_PRICE_PER_MILLION_INPUT = 2.5
GPT_PRICE_PER_MILLION_OUTPUT = 10.0
GPT_54_PRICE_PER_MILLION_INPUT = 2.5
GPT_54_PRICE_PER_MILLION_OUTPUT = 15.0
GPT_54_MINI_PRICE_PER_MILLION_INPUT = 0.75
GPT_54_MINI_PRICE_PER_MILLION_OUTPUT = 4.5
# OpenRouter GPT-4o pricing (ref: https://openrouter.ai/openai/gpt-4o)
OPENROUTER_GPT4O_PRICE_PER_MILLION_INPUT = 2.5
OPENROUTER_GPT4O_PRICE_PER_MILLION_OUTPUT = 10.0

# Maximum input tokens per API embedding call (default; overridable via env var).
# Includes a safety margin to prevent batch failures from provider/model limit differences.
EMBEDDING_API_MAX_INPUT_TOKENS = 8_000

# === Document Tree Node Array Names ===
NODE_ARRAYS = ("texts", "tables", "pictures", "groups", "key_value_items", "form_items")

# === Filtering Thresholds ===
# Short header + empty text_span filtering threshold (in words).
# Headers with empty text_span and text shorter than this length are excluded to reduce invalid candidates.
MIN_HEADER_TEXT_LEN: int = 15

# Overly long text_span filtering threshold (in words).
# Divergent spans from flat sections are unsuitable for embedding; 3200 words ~ 4K tokens.
MAX_SPAN_WORDS: int = 3_200

# === Dataset Label Filename Prefix ===
# The pdfs dataset uses the legacy "10k_" prefix; other datasets default to the dataset name.
DATASET_LABEL_PREFIX = {"pdfs": "10k"}


def get_label_prefix(dataset: str) -> str:
    """Get the label filename prefix for the given dataset."""
    return DATASET_LABEL_PREFIX.get(dataset, dataset)
