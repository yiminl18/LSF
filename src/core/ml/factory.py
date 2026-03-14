# -*- coding: utf-8 -*-
"""
Classifier Factory

Creates classifier instances based on model_type.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

from core.ml.base import BaseClassifier


def _get_xgboost_classifier() -> Type[BaseClassifier]:
    from core.ml.xgboost_model import XGBoostClassifier

    return XGBoostClassifier


def _get_hnn_classifier() -> Type[BaseClassifier]:
    from core.ml.hnn_model import HybridNNClassifier

    return HybridNNClassifier


_CLASSIFIER_REGISTRY: Dict[str, Callable[[], Type[BaseClassifier]]] = {
    "xgb": _get_xgboost_classifier,
    "hnn": _get_hnn_classifier,
}


def register_classifier(
    prefix: str, cls_getter: Callable[[], Type[BaseClassifier]]
) -> None:
    """
    Register a new classifier type.

    Args:
        prefix: Classifier prefix (e.g. 'xgb')
        cls_getter: Function that returns the classifier class
    """
    _CLASSIFIER_REGISTRY[prefix] = cls_getter


def get_classifier_class(model_type: str) -> Type[BaseClassifier]:
    """
    Get classifier class by model_type.

    Args:
        model_type: e.g. 'xgb-sem-struc-v5'

    Returns:
        Classifier class
    """
    prefix = model_type.split("-")[0]
    if prefix not in _CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier type: {prefix}. Available: {list(_CLASSIFIER_REGISTRY.keys())}"
        )
    return _CLASSIFIER_REGISTRY[prefix]()


def _resolve_torch_device(device_str: str) -> "torch.device":
    """Resolve device string to torch.device (auto -> cuda > mps > cpu)"""
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_classifier(
    model_type: str,
    params: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> BaseClassifier:
    """
    Create a classifier instance.

    Args:
        model_type: e.g. 'xgb-sem-struc-v5'
        params: Custom parameters
        feature_names: List of feature names
        device: Neural network device (auto/cpu/cuda/mps), only for PyTorch classifiers

    Returns:
        Classifier instance
    """
    cls = get_classifier_class(model_type)
    clf = cls(params=params, feature_names=feature_names)
    if device and hasattr(clf, "device"):
        clf.device = _resolve_torch_device(device)
    return clf


def load_classifier(path: str, device: Optional[str] = None) -> BaseClassifier:
    """
    Load a classifier from file (auto-detect type).

    Determines classifier type via metadata file; falls back to XGBoost
    for legacy models without classifier_type field.

    Args:
        path: Model file path

    Returns:
        Classifier instance
    """
    load_path = Path(path)
    meta_path = load_path.with_suffix(".meta.json")

    # Default to XGBoost (backward compatible with legacy models)
    classifier_type = "xgb"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classifier_type = meta.get("classifier_type", "xgb")

    cls = _CLASSIFIER_REGISTRY.get(classifier_type, _get_xgboost_classifier)()
    classifier = cls()
    # Set device before load to avoid state inconsistency from initializing
    # on one device (e.g. mps) then switching to another.
    if device and hasattr(classifier, "device"):
        classifier.device = _resolve_torch_device(device)
    classifier.load(path)
    return classifier
