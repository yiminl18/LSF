"""Linear fusion strategy for combining bank score and baseline similarity."""

import numpy as np
from typing import Dict, Any


class LinearFusion:
    """Linear weighted fusion of bank_score and baseline_sim."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 1.0):
        """
        Initialize linear fusion weights.
        
        Args:
            alpha: Weight for bank_score
            beta: Weight for baseline_sim
        """
        self.alpha = alpha
        self.beta = beta
    
    def score(
        self,
        bank_scores: np.ndarray,
        baseline_sims: np.ndarray,
    ) -> np.ndarray:
        """
        Compute final ranking scores via linear combination.
        
        Args:
            bank_scores: Provenance bank similarity scores (normalized)
            baseline_sims: Baseline question-node similarity (normalized)
            
        Returns:
            Final ranking scores
        """
        # Normalize inputs to [0, 1]
        bank_norm = self._minmax_normalize(bank_scores)
        baseline_norm = self._minmax_normalize(baseline_sims)
        
        # Linear combination
        final = self.alpha * bank_norm + self.beta * baseline_norm
        return final
    
    def _minmax_normalize(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        if x.size == 0:
            return x
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float64)
        return (x - lo) / (hi - lo)


