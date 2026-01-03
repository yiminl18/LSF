"""Base classes for feature extraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class HeaderNode:
    """Header node representation."""
    idx_in_texts: int
    text: str
    text_span: str
    page_no: int
    font_size: float
    is_bold: int
    
    @property
    def combined_text(self) -> str:
        """Combined text used for embedding/judging."""
        return f"{self.text} {self.text_span}".strip()


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Feature extractor name (for registry and logging)."""
        pass
    
    @abstractmethod
    def extract_node_features(
        self, 
        headers: List[HeaderNode], 
        context: Any
    ) -> List[Dict[str, float]]:
        """
        Extract features for all nodes given document context.
        
        Args:
            headers: List of header nodes in document order
            context: Pre-computed document-level context
            
        Returns:
            List of feature dicts (one per header)
        """
        pass
    
    @abstractmethod
    def extract_pair_features(
        self,
        node_a: HeaderNode,
        node_b: HeaderNode,
        feat_a: Dict[str, float],
        feat_b: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Extract features for a pair of nodes.
        
        Args:
            node_a, node_b: The two nodes
            feat_a, feat_b: Pre-extracted node-level features
            **kwargs: Additional context (e.g., sim_node_node)
            
        Returns:
            Dict of pair-level features
        """
        pass
    
    @abstractmethod
    def get_pair_feature_names(self) -> List[str]:
        """Return list of pair feature names (for DataFrame columns)."""
        pass
    
    def build_context(self, headers: List[HeaderNode]) -> Any:
        """
        Optional: Build document-level context for feature extraction.
        Default implementation returns None.
        """
        return None


