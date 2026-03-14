# -*- coding: utf-8 -*-
"""
ML module — cross-document node similarity via machine learning.

Provides binary classifiers (XGBoost, HNN) to determine whether two document
headers share the same semantic role.

Components:
- config.py: model configuration (model types, feature modes)
- features.py: feature extraction (F1, F2, F3, etc.)
- dataset.py: dataset construction and positive/negative sample generation
- base.py: classifier abstract base class
- factory.py: classifier factory
"""

from core.ml.config import (
    MODEL_TYPE_TO_MODE,
    XGB_MODEL_TYPES,
    ML_MODEL_TYPES,
    ALL_MODEL_TYPES,
    FEATURE_MODES,
    DEFAULT_MODEL_TYPES,
    CLASSIFIER_PREFIXES,
    get_mode_for_model_type,
    model_type_from_mode,
    get_classifier_prefix,
)
from core.ml.features import extract_ml_features
from core.ml.metrics import compute_mrr_for_eval
from core.ml.dataset import (
    SimilarityDataset,
    ProvenanceAnnotation,
    get_cached_embeddings,
    clear_embeddings_cache,
)
from core.ml.base import BaseClassifier
from core.ml.xgboost_model import XGBoostClassifier
from core.ml.factory import create_classifier, load_classifier, register_classifier

__all__ = [
    # config
    "MODEL_TYPE_TO_MODE",
    "XGB_MODEL_TYPES",
    "ML_MODEL_TYPES",
    "ALL_MODEL_TYPES",
    "FEATURE_MODES",
    "DEFAULT_MODEL_TYPES",
    "CLASSIFIER_PREFIXES",
    "get_mode_for_model_type",
    "model_type_from_mode",
    "get_classifier_prefix",
    # features
    "extract_ml_features",
    # metrics
    "compute_mrr_for_eval",
    # dataset
    "SimilarityDataset",
    "ProvenanceAnnotation",
    "get_cached_embeddings",
    "clear_embeddings_cache",
    # base & implementations
    "BaseClassifier",
    "XGBoostClassifier",
    # factory
    "create_classifier",
    "load_classifier",
    "register_classifier",
]
