"""
V5-only dataset construction and sample generation for the artifact surface.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.embed.embeddings import (
    build_header_embedding_matrix,
    cosine_sim,
    cosine_sim_batch,
    load_document_embeddings,
)
from core.ml.features import RETAINED_MODE, extract_ml_features
from core.ml.lexical import (
    BM25Scorer,
    batch_char3_jaccard,
    batch_fuzzy_ratio,
    batch_token_jaccard,
    batch_token_precision,
    batch_token_recall,
)
from core.doc.feature_extract import (
    DocumentContext,
    HeaderNode,
    build_document_context,
    iter_section_headers,
)

logger = logging.getLogger(__name__)

_EMBEDDINGS_CACHE: Dict[str, Dict[str, List[float]]] = {}
_DOCUMENT_CACHE: Dict[str, Tuple[List[HeaderNode], DocumentContext]] = {}
MAX_CACHE_SIZE = 20


def _ensure_cache_space(cache_dict: Dict, max_size: int = MAX_CACHE_SIZE) -> None:
    if len(cache_dict) >= max_size:
        try:
            del cache_dict[next(iter(cache_dict))]
        except StopIteration:
            pass


def get_cached_embeddings(
    merged_path: str,
    embeddings_dir: str,
    provider: str = "openai",
) -> Dict[str, List[float]]:
    cache_key = merged_path
    if cache_key not in _EMBEDDINGS_CACHE:
        _ensure_cache_space(_EMBEDDINGS_CACHE)
        embeddings, _ = load_document_embeddings(
            merged_path,
            embeddings_dir,
            provider=provider,
        )
        _EMBEDDINGS_CACHE[cache_key] = embeddings
    else:
        value = _EMBEDDINGS_CACHE.pop(cache_key)
        _EMBEDDINGS_CACHE[cache_key] = value
    return _EMBEDDINGS_CACHE[cache_key]


def get_cached_document_structure(
    merged_path: Path,
) -> Tuple[List[HeaderNode], DocumentContext]:
    key = str(merged_path)
    if key not in _DOCUMENT_CACHE:
        _ensure_cache_space(_DOCUMENT_CACHE)
        with open(merged_path, "r", encoding="utf-8") as handle:
            merged_data = json.load(handle)
        header_list = iter_section_headers(merged_data)
        context = build_document_context(header_list)
        _DOCUMENT_CACHE[key] = (header_list, context)
    else:
        value = _DOCUMENT_CACHE.pop(key)
        _DOCUMENT_CACHE[key] = value
    return _DOCUMENT_CACHE[key]


def clear_embeddings_cache() -> None:
    _EMBEDDINGS_CACHE.clear()


def clear_document_cache() -> None:
    _DOCUMENT_CACHE.clear()


@dataclass
class ProvenanceAnnotation:
    doc_id: str
    question: str
    answer: str
    refined_provenance: Optional[List[dict]] = None
    weight: float = 1.0


@dataclass
class DocumentData:
    doc_id: str
    header_list: List[HeaderNode]
    context: DocumentContext
    embeddings: Dict[str, List[float]]
    provenance_indices: set[int] = field(default_factory=set)
    weight: float = 1.0


class SimilarityDataset:
    """Generate the retained mode-25 training samples."""

    def __init__(
        self,
        annotations: List[ProvenanceAnnotation],
        merged_json_dir: str,
        embeddings_dir: str,
        query: str,
        tree_embeddings_dir: Optional[str] = None,
        neg_ratio: int = 5,
        hard_neg_ratio: float = 0.6,
        feature_mode: int = RETAINED_MODE,
        provider: str = "openai",
        proximal_neg: bool = False,
        curriculum: bool = False,
        curriculum_phase: str = "easy",
        q_idx: Optional[int] = None,
    ):
        if feature_mode != RETAINED_MODE:
            raise ValueError(
                f"Artifact only supports feature_mode={RETAINED_MODE}, got {feature_mode}"
            )

        self.annotations = annotations
        self.merged_json_dir = Path(merged_json_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.tree_embeddings_dir = Path(tree_embeddings_dir) if tree_embeddings_dir else None
        self.query = query
        self.neg_ratio = neg_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.feature_mode = feature_mode
        self.provider = provider
        self.proximal_neg = proximal_neg
        self.curriculum = curriculum
        self.curriculum_phase = curriculum_phase

        self._annotation_by_doc_id: Dict[str, ProvenanceAnnotation] = {}
        for ann in annotations:
            self._annotation_by_doc_id.setdefault(ann.doc_id, ann)

        self.documents: Dict[str, DocumentData] = {}
        self._documents_loaded = False
        self.query_embedding: Optional[List[float]] = None
        self._bm25_cache: Dict[str, BM25Scorer] = {}
        self._bm25_query_scores_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._lexical_query_scores_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _resolve_structure_path(self, doc_id: str) -> Path:
        return self.merged_json_dir / f"{doc_id}_reconstructed.json"

    def _load_document(self, doc_id: str) -> None:
        if doc_id in self.documents:
            return

        ann = self._annotation_by_doc_id.get(doc_id)
        if ann is None:
            return

        processing_path = self._resolve_structure_path(ann.doc_id)
        if not processing_path.exists():
            logger.warning("Document structure not found: %s", processing_path)
            return

        header_list, context = get_cached_document_structure(processing_path)
        embeddings = get_cached_embeddings(
            str(processing_path),
            str(self.embeddings_dir),
            provider=self.provider,
        )

        text_idx_to_list_idx = {h.idx_in_texts: i for i, h in enumerate(header_list)}
        provenance_indices = set()
        if ann.refined_provenance:
            for item in ann.refined_provenance:
                header_idx = item.get("header_idx")
                if header_idx is None:
                    continue
                list_idx = text_idx_to_list_idx.get(header_idx)
                if list_idx is not None:
                    provenance_indices.add(list_idx)

        self.documents[ann.doc_id] = DocumentData(
            doc_id=ann.doc_id,
            header_list=header_list,
            context=context,
            embeddings=embeddings,
            provenance_indices=provenance_indices,
            weight=ann.weight,
        )

    def _load_documents(self) -> None:
        if self._documents_loaded:
            return
        for doc_id in self._annotation_by_doc_id:
            self._load_document(doc_id)
        self._documents_loaded = True

    def set_query_embedding(self, embedding: List[float]) -> None:
        self.query_embedding = embedding

    def _get_or_build_bm25_scores(
        self,
        doc_id: str,
        doc: DocumentData,
    ) -> Tuple[BM25Scorer, np.ndarray, np.ndarray]:
        scorer = self._bm25_cache.get(doc_id)
        if scorer is None:
            scorer = BM25Scorer([header.combined_text for header in doc.header_list])
            self._bm25_cache[doc_id] = scorer

        cached = self._bm25_query_scores_cache.get(doc_id)
        if cached is None:
            header_scores = scorer.score_batch(
                self.query,
                [header.text or "" for header in doc.header_list],
            )
            combined_scores = scorer.score_batch(
                self.query,
                [header.combined_text or "" for header in doc.header_list],
            )
            cached = (header_scores, combined_scores)
            self._bm25_query_scores_cache[doc_id] = cached
        return scorer, cached[0], cached[1]

    def _get_or_build_lexical_scores(
        self,
        doc_id: str,
        doc: DocumentData,
    ) -> Dict[str, np.ndarray]:
        cached = self._lexical_query_scores_cache.get(doc_id)
        if cached is not None:
            return cached

        texts = [header.text or "" for header in doc.header_list]
        paths = [header.processing_path or "" for header in doc.header_list]
        cached = {
            "tok_jaccard": batch_token_jaccard(self.query, texts),
            "tok_recall": batch_token_recall(self.query, texts),
            "tok_precision": batch_token_precision(self.query, texts),
            "char3_jaccard": batch_char3_jaccard(self.query, texts),
            "fuzz_ratio": batch_fuzzy_ratio(self.query, texts),
            "path_tok_jaccard": batch_token_jaccard(self.query, paths),
        }
        self._lexical_query_scores_cache[doc_id] = cached
        return cached

    def generate_samples(
        self,
        seed: int = 42,
        feature_subset: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, List[Dict], List[str]]:
        random.seed(seed)
        np.random.seed(seed)

        if self.curriculum:
            self.hard_neg_ratio = 0.2 if self.curriculum_phase == "easy" else 0.8

        self._load_documents()
        doc_ids = list(self.documents.keys())
        if len(doc_ids) < 2:
            return np.array([]), np.array([]), [], [], np.array([]), [], []

        refined_by_doc: Dict[str, Dict[int, dict]] = {}
        for ann in self.annotations:
            if ann.refined_provenance:
                refined_by_doc[ann.doc_id] = {
                    item["header_idx"]: item for item in ann.refined_provenance
                }

        q_vec = (
            np.asarray(self.query_embedding, dtype=np.float32)
            if self.query_embedding is not None
            else None
        )

        doc_query_sims: Dict[str, np.ndarray] = {}
        doc_path_query_sims: Dict[str, np.ndarray] = {}
        doc_bm25: Dict[str, BM25Scorer] = {}
        doc_bm25_header_scores: Dict[str, np.ndarray] = {}
        doc_bm25_combined_scores: Dict[str, np.ndarray] = {}
        doc_lexical: Dict[str, Dict[str, np.ndarray]] = {}
        non_prov_by_doc: Dict[str, List[int]] = {}
        proximal_candidates: Dict[str, List[int]] = {}
        structural_candidates: Dict[str, Dict[int, List[int]]] = {}

        for doc_id, doc in self.documents.items():
            n_headers = len(doc.header_list)
            dim = len(q_vec) if q_vec is not None else 0
            mat, valid, path_mat, path_valid = build_header_embedding_matrix(
                doc.header_list,
                doc.embeddings,
                dim,
            )
            if q_vec is not None and dim > 0 and n_headers > 0:
                query_sims = cosine_sim_batch(mat, q_vec)
                query_sims[~valid] = 0.0
                path_query_sims = cosine_sim_batch(path_mat, q_vec)
                path_query_sims[~path_valid] = 0.0
            else:
                query_sims = np.zeros(n_headers, dtype=np.float32)
                path_query_sims = np.zeros(n_headers, dtype=np.float32)
            doc_query_sims[doc_id] = query_sims
            doc_path_query_sims[doc_id] = path_query_sims

            scorer, bm25_header, bm25_combined = self._get_or_build_bm25_scores(doc_id, doc)
            doc_bm25[doc_id] = scorer
            doc_bm25_header_scores[doc_id] = bm25_header
            doc_bm25_combined_scores[doc_id] = bm25_combined
            doc_lexical[doc_id] = self._get_or_build_lexical_scores(doc_id, doc)

            non_prov = [
                idx for idx in range(n_headers) if idx not in doc.provenance_indices
            ]
            non_prov_by_doc[doc_id] = non_prov
            if self.proximal_neg:
                proximal_candidates[doc_id] = sorted(
                    non_prov,
                    key=lambda idx, sims=query_sims: sims[idx],
                    reverse=True,
                )
                level_map: Dict[int, List[int]] = {}
                for idx in non_prov:
                    level = doc.header_list[idx].structure_level
                    level_map.setdefault(level, []).append(idx)
                structural_candidates[doc_id] = level_map

        def apply_visual_injection(doc_id: str, header_idx: int, header: HeaderNode) -> HeaderNode:
            refined_map = refined_by_doc.get(doc_id)
            if not refined_map:
                return header
            item = refined_map.get(header.idx_in_texts)
            if not item or "visuals" not in item:
                return header
            visuals = item["visuals"]
            return replace(
                header,
                font_size=visuals.get("font_size", header.font_size),
                is_bold=visuals.get("is_bold", header.is_bold),
            )

        def build_feature_kwargs(
            doc_id_a: str,
            idx_a: int,
            doc_id_b: str,
            idx_b: int,
        ) -> Dict[str, float]:
            doc_a = self.documents[doc_id_a]
            doc_b = self.documents[doc_id_b]
            header_a = doc_a.header_list[idx_a]
            header_b = doc_b.header_list[idx_b]
            a_emb = doc_a.embeddings.get(header_a.combined_text)
            b_emb = doc_b.embeddings.get(header_b.combined_text)
            lexical = doc_lexical[doc_id_b]
            return {
                "_pre_f1": float(cosine_sim(a_emb, b_emb) if a_emb and b_emb else 0.0),
                "_pre_sim_a": float(doc_query_sims[doc_id_a][idx_a]),
                "_pre_sim_b": float(doc_query_sims[doc_id_b][idx_b]),
                "_pre_sim_path_a": float(doc_path_query_sims[doc_id_a][idx_a]),
                "_pre_sim_path_b": float(doc_path_query_sims[doc_id_b][idx_b]),
                "_pre_bm25_header_b": float(doc_bm25_header_scores[doc_id_b][idx_b]),
                "_pre_bm25_combined_b": float(doc_bm25_combined_scores[doc_id_b][idx_b]),
                "_pre_q_b_tok_jaccard": float(lexical["tok_jaccard"][idx_b]),
                "_pre_q_b_tok_recall": float(lexical["tok_recall"][idx_b]),
                "_pre_q_b_tok_precision": float(lexical["tok_precision"][idx_b]),
                "_pre_q_b_char3_jaccard": float(lexical["char3_jaccard"][idx_b]),
                "_pre_q_b_fuzz_ratio": float(lexical["fuzz_ratio"][idx_b]),
                "_pre_q_path_b_tok_jaccard": float(lexical["path_tok_jaccard"][idx_b]),
            }

        def append_sample(
            label: int,
            doc_id_a: str,
            idx_a: int,
            doc_id_b: str,
            idx_b: int,
            weight: float,
            final_samples: List[Dict],
        ) -> None:
            doc_a = self.documents[doc_id_a]
            doc_b = self.documents[doc_id_b]
            header_a = apply_visual_injection(doc_id_a, idx_a, doc_a.header_list[idx_a])
            header_b = apply_visual_injection(doc_id_b, idx_b, doc_b.header_list[idx_b])
            features = extract_ml_features(
                header_a,
                header_b,
                idx_a,
                idx_b,
                doc_a.context,
                doc_b.context,
                doc_a.embeddings,
                doc_b.embeddings,
                self.query_embedding,
                query_text=self.query,
                mode=self.feature_mode,
                bm25_scorer_b=doc_bm25.get(doc_id_b),
                **build_feature_kwargs(doc_id_a, idx_a, doc_id_b, idx_b),
            )
            final_samples.append(
                {
                    "label": label,
                    "doc_pair": f"{doc_id_a}_{doc_id_b}",
                    "group_key": f"{doc_id_a}:{idx_a}",
                    "meta": {
                        "doc_a": doc_id_a,
                        "doc_b": doc_id_b,
                        "idx_a": idx_a,
                        "idx_b": idx_b,
                    },
                    "features": features,
                    "weight": weight,
                }
            )

        final_samples: List[Dict] = []

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a = self.documents[doc_ids[i]]
                doc_b = self.documents[doc_ids[j]]
                if not doc_a.provenance_indices or not doc_b.provenance_indices:
                    continue
                for idx_a in doc_a.provenance_indices:
                    for idx_b in doc_b.provenance_indices:
                        append_sample(
                            1,
                            doc_ids[i],
                            idx_a,
                            doc_ids[j],
                            idx_b,
                            min(doc_a.weight, doc_b.weight),
                            final_samples,
                        )

        n_positive = len(final_samples)
        n_negative_target = n_positive * self.neg_ratio
        n_hard_neg = int(n_negative_target * self.hard_neg_ratio)
        n_easy_neg = n_negative_target - n_hard_neg

        hard_neg_count = 0
        attempts = 0
        max_attempts = max(n_hard_neg * 10, 1)
        while hard_neg_count < n_hard_neg and attempts < max_attempts:
            attempts += 1
            doc_id_a = random.choice(doc_ids)
            doc_a = self.documents[doc_id_a]
            if not doc_a.provenance_indices:
                continue
            idx_a = random.choice(list(doc_a.provenance_indices))

            doc_id_b = random.choice(doc_ids)
            if doc_id_b == doc_id_a:
                continue
            non_prov = non_prov_by_doc.get(doc_id_b, [])
            if not non_prov:
                continue

            if self.proximal_neg:
                draw = random.random()
                if draw < 0.4 and proximal_candidates.get(doc_id_b):
                    top_k = max(3, len(proximal_candidates[doc_id_b]) // 4)
                    idx_b = random.choice(proximal_candidates[doc_id_b][:top_k])
                elif draw < 0.6:
                    level = doc_a.header_list[idx_a].structure_level
                    candidates = structural_candidates.get(doc_id_b, {}).get(level, [])
                    idx_b = random.choice(candidates or non_prov)
                else:
                    idx_b = random.choice(non_prov)
            else:
                idx_b = random.choice(non_prov)

            append_sample(0, doc_id_a, idx_a, doc_id_b, idx_b, doc_a.weight, final_samples)
            hard_neg_count += 1

        easy_neg_count = 0
        all_non_prov_pairs = [
            (doc_id, idx)
            for doc_id, indices in non_prov_by_doc.items()
            for idx in indices
        ]
        attempts = 0
        max_attempts = max(n_easy_neg * 5, 1)
        while (
            easy_neg_count < n_easy_neg
            and len(all_non_prov_pairs) >= 2
            and attempts < max_attempts
        ):
            attempts += 1
            (doc_id_a, idx_a), (doc_id_b, idx_b) = random.sample(all_non_prov_pairs, 2)
            if doc_id_a == doc_id_b:
                continue
            append_sample(0, doc_id_a, idx_a, doc_id_b, idx_b, 1.0, final_samples)
            easy_neg_count += 1

        if not final_samples:
            return np.array([]), np.array([]), [], [], np.array([]), [], []

        all_feature_names = sorted(final_samples[0]["features"].keys())
        feature_names = (
            [name for name in all_feature_names if name in feature_subset]
            if feature_subset is not None
            else all_feature_names
        )

        X_rows = [[sample["features"].get(name, 0.0) for name in feature_names] for sample in final_samples]
        y_rows = [sample["label"] for sample in final_samples]
        doc_pair_rows = [sample["doc_pair"] for sample in final_samples]
        weight_rows = [sample.get("weight", 1.0) for sample in final_samples]
        meta_rows = [sample["meta"] for sample in final_samples]
        group_key_rows = [sample.get("group_key", "") for sample in final_samples]

        return (
            np.asarray(X_rows, dtype=np.float32),
            np.asarray(y_rows, dtype=np.int64),
            feature_names,
            doc_pair_rows,
            np.asarray(weight_rows, dtype=np.float32),
            meta_rows,
            group_key_rows,
        )
