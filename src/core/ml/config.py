"""
Artifact ML configuration for the retained v5-only model surface.
"""

from typing import Dict, FrozenSet, List

MODEL_TYPE_TO_MODE: Dict[str, int] = {
    "xgb-sem-struc-v5": 25,
    "hnn-sem-struc-v5": 25,
}

CLASSIFIER_PREFIXES: List[str] = ["xgb", "hnn"]
XGB_MODEL_TYPES: List[str] = [k for k in MODEL_TYPE_TO_MODE if k.startswith("xgb-")]
HNN_MODEL_TYPES: List[str] = [k for k in MODEL_TYPE_TO_MODE if k.startswith("hnn-")]
ML_MODEL_TYPES: List[str] = XGB_MODEL_TYPES + HNN_MODEL_TYPES
ALL_MODEL_TYPES: List[str] = ML_MODEL_TYPES + ["rag"]
FEATURE_MODES: List[int] = [25]
DEFAULT_MODEL_TYPES: List[str] = ["xgb-sem-struc-v5"]
DEFAULT_SEEDS: List[int] = [41, 42, 43]

CAP_SIM_B = "sim_b"
CAP_STRUC = "struc"
CAP_VISUAL2 = "visual2"
CAP_CONTENT = "content"
CAP_LEXICAL = "lexical"
CAP_BM25 = "bm25"

MODE_CAPS: Dict[int, FrozenSet[str]] = {
    25: frozenset(
        {CAP_SIM_B, CAP_STRUC, CAP_VISUAL2, CAP_CONTENT, CAP_LEXICAL, CAP_BM25}
    )
}


def get_mode_for_model_type(model_type: str) -> int:
    """Return the retained feature mode for a model type."""
    if model_type not in MODEL_TYPE_TO_MODE:
        raise ValueError(f"Unknown model type: {model_type}. Valid: {ML_MODEL_TYPES}")
    return MODEL_TYPE_TO_MODE[model_type]


def get_classifier_prefix(model_type: str) -> str:
    """Return the classifier prefix for a retained model type."""
    return model_type.split("-")[0]


def model_type_from_mode(mode: int, prefix: str = "xgb") -> str:
    """Return the retained model type for the retained mode and classifier prefix."""
    model_type = f"{prefix}-sem-struc-v5"
    if mode != 25 or model_type not in MODEL_TYPE_TO_MODE:
        raise ValueError(
            f"Artifact only supports mode 25 with retained v5 models, got mode={mode}, prefix={prefix}"
        )
    return model_type


def mode_has(mode: int, cap: str) -> bool:
    """Return whether the retained mode exposes a capability."""
    return mode == 25 and cap in MODE_CAPS[25]


__all__ = [
    "MODEL_TYPE_TO_MODE",
    "CLASSIFIER_PREFIXES",
    "XGB_MODEL_TYPES",
    "HNN_MODEL_TYPES",
    "ML_MODEL_TYPES",
    "ALL_MODEL_TYPES",
    "FEATURE_MODES",
    "DEFAULT_MODEL_TYPES",
    "DEFAULT_SEEDS",
    "CAP_SIM_B",
    "CAP_STRUC",
    "CAP_VISUAL2",
    "CAP_CONTENT",
    "CAP_LEXICAL",
    "CAP_BM25",
    "MODE_CAPS",
    "get_mode_for_model_type",
    "model_type_from_mode",
    "get_classifier_prefix",
    "mode_has",
]
