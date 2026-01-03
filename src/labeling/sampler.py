"""Positive and negative pair sampling for similarity model training."""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class SimilarityPairSampler:
    """Sample positive and hard negative pairs for training."""
    
    def __init__(
        self,
        pos_within_mode: str = "all",      # all | sample
        pos_cross_mode: str = "all",       # all | sample | none
        neg_within_per_prov: int = 5,
        neg_cross_per_prov: int = 5,
        cross_neg_pattern_bucket: bool = True,
        seed: int = 7,
    ):
        """
        Initialize pair sampler.
        
        Args:
            pos_within_mode: How to sample within-PDF positive pairs
            pos_cross_mode: How to sample cross-PDF positive pairs
            neg_within_per_prov: Number of within-PDF negatives per prov node
            neg_cross_per_prov: Number of cross-PDF negatives per prov node
            cross_neg_pattern_bucket: Use pattern bucket for cross-PDF negatives
            seed: Random seed
        """
        self.pos_within_mode = pos_within_mode
        self.pos_cross_mode = pos_cross_mode
        self.neg_within_per_prov = neg_within_per_prov
        self.neg_cross_per_prov = neg_cross_per_prov
        self.cross_neg_pattern_bucket = cross_neg_pattern_bucket
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def sample_pairs_for_question(
        self,
        question_index: int,
        training_entries: List[Dict[str, Any]],
        feature_extractor,
        embedding_cache: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Sample pairs for a single question.
        
        Args:
            question_index: Question index
            training_entries: List of training entries from per_question_training.jsonl
            feature_extractor: Feature extractor instance (from src.features)
            embedding_cache: Dict mapping node_text -> embedding vector
            
        Returns:
            List of pair dicts with features and labels
        """
        # Group by PDF and separate prov/nonprov
        pdf_groups = defaultdict(lambda: {'prov': [], 'nonprov': []})
        
        for entry in training_entries:
            if entry.get('question_index') != question_index:
                continue
            
            pdf_id = entry['pdf_id']
            if entry.get('label') == 1:
                pdf_groups[pdf_id]['prov'].append(entry)
            else:
                pdf_groups[pdf_id]['nonprov'].append(entry)
        
        pairs = []
        
        # 1. Within-PDF positive pairs
        pairs.extend(self._sample_within_pdf_positives(pdf_groups))
        
        # 2. Cross-PDF positive pairs
        pairs.extend(self._sample_cross_pdf_positives(pdf_groups))
        
        # 3. Within-PDF negative pairs
        pairs.extend(self._sample_within_pdf_negatives(pdf_groups, embedding_cache))
        
        # 4. Cross-PDF negative pairs
        pairs.extend(self._sample_cross_pdf_negatives(pdf_groups, embedding_cache))
        
        # Extract features for all pairs
        pairs_with_features = []
        for pair in pairs:
            # TODO: Call feature_extractor.extract_pair_features()
            # This requires node objects, not just dicts
            # In practice, you'd reconstruct HeaderNode from entry dict
            pair_features = self._extract_pair_features_stub(pair, embedding_cache)
            pairs_with_features.append({
                'question_index': question_index,
                'label': pair['label'],
                **pair_features,
            })
        
        return pairs_with_features
    
    def _sample_within_pdf_positives(
        self,
        pdf_groups: Dict[str, Dict[str, List[Dict]]]
    ) -> List[Dict]:
        """Sample within-PDF positive pairs (prov, prov)."""
        pairs = []
        for pdf_id, group in pdf_groups.items():
            provs = group['prov']
            if len(provs) < 2:
                continue
            
            # All pairs or sample
            if self.pos_within_mode == "all":
                for i, p1 in enumerate(provs):
                    for p2 in provs[i+1:]:
                        pairs.append({
                            'node_a': p1,
                            'node_b': p2,
                            'label': 1,
                            'pair_type': 'within_pos',
                        })
        
        return pairs
    
    def _sample_cross_pdf_positives(
        self,
        pdf_groups: Dict[str, Dict[str, List[Dict]]]
    ) -> List[Dict]:
        """Sample cross-PDF positive pairs (prov, prov)."""
        pairs = []
        
        if self.pos_cross_mode == "none":
            return pairs
        
        # Collect all prov nodes
        all_provs = []
        for pdf_id, group in pdf_groups.items():
            for prov in group['prov']:
                all_provs.append((pdf_id, prov))
        
        if self.pos_cross_mode == "all":
            for i, (pdf1, p1) in enumerate(all_provs):
                for pdf2, p2 in all_provs[i+1:]:
                    if pdf1 != pdf2:
                        pairs.append({
                            'node_a': p1,
                            'node_b': p2,
                            'label': 1,
                            'pair_type': 'cross_pos',
                        })
        
        return pairs
    
    def _sample_within_pdf_negatives(
        self,
        pdf_groups: Dict[str, Dict[str, List[Dict]]],
        embedding_cache: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """Sample within-PDF hard negatives (prov, nonprov)."""
        pairs = []
        
        for pdf_id, group in pdf_groups.items():
            provs = group['prov']
            nonprovs = group['nonprov']
            
            if not provs or not nonprovs:
                continue
            
            for prov in provs:
                # Select top-K nonprovs by embedding similarity
                prov_emb = embedding_cache.get(prov['text'])
                if prov_emb is None:
                    continue
                
                scored = []
                for nonprov in nonprovs:
                    nonprov_emb = embedding_cache.get(nonprov['text'])
                    if nonprov_emb is not None:
                        sim = self._cosine_sim(prov_emb, nonprov_emb)
                        scored.append((sim, nonprov))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                top_k = scored[:self.neg_within_per_prov]
                
                for _, nonprov in top_k:
                    pairs.append({
                        'node_a': prov,
                        'node_b': nonprov,
                        'label': 0,
                        'pair_type': 'within_neg',
                    })
        
        return pairs
    
    def _sample_cross_pdf_negatives(
        self,
        pdf_groups: Dict[str, Dict[str, List[Dict]]],
        embedding_cache: Dict[str, np.ndarray],
    ) -> List[Dict]:
        """Sample cross-PDF hard negatives (prov, nonprov)."""
        pairs = []
        
        # Collect all prov and nonprov across PDFs
        all_provs = []
        all_nonprovs = []
        
        for pdf_id, group in pdf_groups.items():
            for prov in group['prov']:
                all_provs.append((pdf_id, prov))
            for nonprov in group['nonprov']:
                all_nonprovs.append((pdf_id, nonprov))
        
        for pdf_prov, prov in all_provs:
            prov_emb = embedding_cache.get(prov['text'])
            if prov_emb is None:
                continue
            
            # Sample from other PDFs
            candidates = [(pdf, np) for pdf, np in all_nonprovs if pdf != pdf_prov]
            
            if not candidates:
                continue
            
            scored = []
            for pdf_nonprov, nonprov in candidates:
                nonprov_emb = embedding_cache.get(nonprov['text'])
                if nonprov_emb is not None:
                    sim = self._cosine_sim(prov_emb, nonprov_emb)
                    scored.append((sim, nonprov))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:self.neg_cross_per_prov]
            
            for _, nonprov in top_k:
                pairs.append({
                    'node_a': prov,
                    'node_b': nonprov,
                    'label': 0,
                    'pair_type': 'cross_neg',
                })
        
        return pairs
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def _extract_pair_features_stub(
        self,
        pair: Dict,
        embedding_cache: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Stub for feature extraction (to be replaced with real implementation)."""
        node_a = pair['node_a']
        node_b = pair['node_b']
        
        # Compute sim_node_node
        emb_a = embedding_cache.get(node_a['text'])
        emb_b = embedding_cache.get(node_b['text'])
        sim = self._cosine_sim(emb_a, emb_b) if (emb_a is not None and emb_b is not None) else 0.0
        
        return {
            'sim_node_node': sim,
            'node_a_text': node_a['text'],
            'node_b_text': node_b['text'],
            'pdf_id_a': node_a.get('pdf_id', ''),
            'pdf_id_b': node_b.get('pdf_id', ''),
            # TODO: Add other 7 features from feature_extractor
        }


