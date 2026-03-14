# -*- coding: utf-8 -*-
"""
LSF Experiment Pipeline Modules

1. preprocess - PDF to Intermediate JSON
2. build_processing_json - Intermediate to Processing JSON
3. generate_embeddings - Generate Vectors
4. generate_labels - Generate Provenance Labels
5. split_dataset - Train/Test Split
6. train_model - Train Ranking Models
7. evaluate_model - Evaluate Models

Uses lazy imports to avoid warnings when running via `python -m`.
"""

__all__ = [
    "preprocess_documents",
    "reconstruct_documents",
    "generate_embeddings",
    "generate_labels",
    "split_dataset",
    "train_models",
    "evaluate_models",
]

# Lazy import mapping
_LAZY_IMPORTS = {
    "preprocess_documents": ("core.pipeline.preprocess", "preprocess_documents"),
    "reconstruct_documents": (
        "core.pipeline.build_processing_json",
        "reconstruct_documents",
    ),
    "generate_embeddings": ("core.pipeline.generate_embeddings", "generate_embeddings"),
    "generate_labels": ("core.pipeline.generate_labels", "generate_labels"),
    "split_dataset": ("core.pipeline.split_dataset", "split_dataset"),
    "train_models": ("core.pipeline.train_model", "train_models"),
    "evaluate_models": ("core.pipeline.evaluate_model", "evaluate_models"),
}


def __getattr__(name: str):
    """Lazy import: only import modules when actually accessed."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
