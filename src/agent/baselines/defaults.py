"""Single source of truth for cross-baseline defaults.

Edit this module when you want to globally swap the reader / OCR / judge
model — the runner, majority-vote harness, and each extractor's default
kwargs all read from here.
"""

from __future__ import annotations

from pathlib import Path

# Reader / agent / judge LLM. Every baseline uses this unless the caller
# passes an explicit override via CLI flag or extract() kwarg.
DEFAULT_LLM_PROVIDER: str = "azure"
DEFAULT_LLM_MODEL: str = "gpt-5.4-mini"

# DeepRead OCR vision call.
DEFAULT_OCR_PROVIDER: str = "azure"
DEFAULT_OCR_MODEL: str = "gpt-5.4-mini"


def resolve_dataset_root(config: dict) -> Path:
    """Normalize the dataset root from an experiment config dict.

    Accepts both shapes used across the codebase:
      - `dataset_root: path/to/<name>/latest` (loader-style)
      - `dataset: <name>` + `data_root: datasets/` (majority-vote-style)
    """
    explicit = config.get("dataset_root")
    if explicit:
        return Path(str(explicit))
    dataset = config.get("dataset")
    if dataset:
        data_root = Path(str(config.get("data_root", "datasets")))
        return data_root / str(dataset) / "latest"
    raise KeyError(
        "config must specify either `dataset_root` or `dataset` (+ optional `data_root`)"
    )
