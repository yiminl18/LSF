"""XGBoost-based similarity model implementation."""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Optional


class XGBoostSimilarityModel:
    """XGBoost classifier for node-pair similarity prediction."""
    
    def __init__(self, **params):
        """
        Initialize XGBoost model with parameters.
        
        Args:
            **params: XGBoost hyperparameters
        """
        self.params = params
        self.model: Optional[xgb.XGBClassifier] = None
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the similarity model.
        
        Args:
            X: Feature matrix (n_pairs, n_features)
            y: Binary labels (n_pairs,)
        """
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict similarity probabilities.
        
        Args:
            X: Feature matrix (n_pairs, n_features)
            
        Returns:
            Probabilities of positive class (n_pairs,)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
    
    def load(self, path: str):
        """Load model from file."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (gain)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.get_booster().get_score(importance_type='gain')
        return importance


