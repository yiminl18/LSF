# -*- coding: utf-8 -*-
"""
XGBoost classifier implementation.

Wraps XGBoost binary classification for cross-document node similarity
prediction. Supports MRR as an early stopping metric.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.ml.base import BaseClassifier
from core.ml.metrics import compute_mrr_for_eval

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostClassifier(BaseClassifier):
    """XGBoost binary classifier."""

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required: pip install xgboost")
        super().__init__(params, feature_names)
        self.model: Optional[xgb.Booster] = None

    @classmethod
    def get_classifier_type(cls) -> str:
        return "xgb"

    def get_default_params(self) -> Dict:
        return {
            "objective": "binary:logistic",
            "scale_pos_weight": 5,
            "max_depth": 4,
            "eta": 0.1,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": ["auc", "aucpr"],
            "seed": 42,
            "nthread": 1,
        }

    def _make_mrr_eval(
        self,
        val_doc_ids: List[str],
    ):
        """Create MRR evaluation function (closure capturing val_doc_ids)."""

        def mrr_eval(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            labels = dtrain.get_label()
            mrr = compute_mrr_for_eval(labels, preds, val_doc_ids)
            return "mrr", mrr

        return mrr_eval

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
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train model with MRR early stopping + group-aware split."""
        seed = self.params.get("seed", 42)

        # Group-aware split (consistent with LambdaMART/HNN)
        if group_keys is not None:
            group_indices = self._build_group_indices(group_keys)
            train_idx, val_idx = self._group_split(group_indices, val_split, seed)
        else:
            n_samples = len(y)
            indices = np.random.permutation(n_samples)
            n_val = int(n_samples * val_split)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        w_train = sample_weight[train_idx] if sample_weight is not None else None
        w_val = sample_weight[val_idx] if sample_weight is not None else None

        # Keep validation doc_ids
        val_doc_ids = None
        if doc_ids is not None:
            val_doc_ids = [doc_ids[i] for i in val_idx]

        dtrain = xgb.DMatrix(
            X_train, label=y_train, weight=w_train, feature_names=self.feature_names
        )
        dval = xgb.DMatrix(
            X_val, label=y_val, weight=w_val, feature_names=self.feature_names
        )

        evals = [(dtrain, "train"), (dval, "val")]

        # Training parameters
        evals_result = {}
        train_kwargs = {
            "params": self.params,
            "dtrain": dtrain,
            "num_boost_round": num_rounds,
            "evals": evals,
            "early_stopping_rounds": early_stopping_rounds,
            "evals_result": evals_result,
            "verbose_eval": verbose,
        }

        # MRR early stopping (requires doc_ids)
        if use_mrr and val_doc_ids:
            train_kwargs["custom_metric"] = self._make_mrr_eval(val_doc_ids)
            train_kwargs["maximize"] = True  # Higher MRR is better

        self.model = xgb.train(**train_kwargs)

        # Return metrics
        metrics = {
            "best_iteration": getattr(
                self.model, "best_iteration", self.model.num_boosted_rounds()
            ),
        }

        # Add AUC metrics if available
        if "auc" in evals_result.get("val", {}):
            metrics["train_auc"] = evals_result["train"]["auc"][-1]
            metrics["val_auc"] = evals_result["val"]["auc"][-1]
        if "aucpr" in evals_result.get("val", {}):
            metrics["train_aucpr"] = evals_result["train"]["aucpr"][-1]
            metrics["val_aucpr"] = evals_result["val"]["aucpr"][-1]

        # Add MRR metrics if used
        if use_mrr and val_doc_ids and "mrr" in evals_result.get("val", {}):
            metrics["train_mrr"] = evals_result["train"]["mrr"][-1]
            metrics["val_mrr"] = evals_result["val"]["mrr"][-1]

        return metrics

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Batch prediction."""
        if self.model is None:
            raise ValueError("Model not trained")
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.get_score(importance_type="gain")

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(save_path))

        # Save metadata (including classifier_type)
        meta_path = save_path.with_suffix(".meta.json")
        meta = {
            "classifier_type": self.get_classifier_type(),
            "params": self.params,
            "feature_names": self.feature_names,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str) -> None:
        """Load model."""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self.model = xgb.Booster()
        self.model.load_model(str(load_path))

        # Load metadata
        meta_path = load_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.params = meta.get("params", self.params)
            self.feature_names = meta.get("feature_names")
