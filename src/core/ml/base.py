# -*- coding: utf-8 -*-
"""
Classifier abstract base class.

Defines a unified interface for all classifiers, enabling extension to
different ML frameworks.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


class BaseClassifier(ABC):
    """
    Abstract base class for classifiers.

    All classifiers (XGBoost, HNN, etc.) must implement this interface.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.params = params if params is not None else self.get_default_params()
        self.feature_names = feature_names
        self.model = None  # Subclasses initialize the concrete model object

    @classmethod
    @abstractmethod
    def get_classifier_type(cls) -> str:
        """Return classifier type identifier (e.g. 'xgb')."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict:
        """Return default parameters."""
        pass

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        doc_ids: Optional[List[str]] = None,
        group_keys: Optional[List[str]] = None,
        num_rounds: int = 100,
        early_stopping_rounds: Optional[int] = 10,
        val_split: float = 0.2,
        verbose: bool = True,
        use_mrr: bool = True,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label array (n_samples,)
            doc_ids: Document IDs per sample (used for MRR computation)
            group_keys: Group key list (for ranking models like LambdaMART)
            num_rounds: Number of training rounds
            early_stopping_rounds: Early stopping patience
            val_split: Validation set ratio
            verbose: Whether to print training info
            use_mrr: Whether to use MRR as early stopping metric (default True)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Batch prediction.

        Args:
            X: Feature matrix (n_samples, n_features)
            **kwargs: Subclass extension parameters (e.g. query_embedding)

        Returns:
            Probability array [0, 1]
        """
        pass

    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Predict a single sample.

        Args:
            features: Feature dictionary

        Returns:
            Probability [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained")
        if self.feature_names is None:
            raise ValueError("Feature names not set")
        X = np.array([[features.get(k, 0.0) for k in self.feature_names]])
        return float(self.predict(X)[0])

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model and metadata."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model and metadata."""
        pass

    # ========== Group-Aware Split Utilities ==========

    @staticmethod
    def _build_group_indices(group_keys: List[str]) -> Dict[str, List[int]]:
        """Build group -> indices mapping from group_keys."""
        group_indices: Dict[str, List[int]] = OrderedDict()
        for i, key in enumerate(group_keys):
            if key not in group_indices:
                group_indices[key] = []
            group_indices[key].append(i)
        return group_indices

    @staticmethod
    def _group_split(
        group_indices: Dict[str, List[int]],
        val_ratio: float,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split train/val by group, preserving group integrity."""
        group_keys = list(group_indices.keys())
        n_groups = len(group_keys)
        n_val_groups = max(1, int(n_groups * val_ratio))

        rng = np.random.default_rng(seed)
        val_group_set = set(rng.choice(n_groups, n_val_groups, replace=False))

        train_idx, val_idx = [], []
        for i, key in enumerate(group_keys):
            indices = group_indices[key]
            if i in val_group_set:
                val_idx.extend(indices)
            else:
                train_idx.extend(indices)

        return np.array(train_idx), np.array(val_idx)
