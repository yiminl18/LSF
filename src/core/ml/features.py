"""
Mode-25 ML feature extraction for the retained artifact surface.

The artifact keeps only one Problem 1 feature surface:
- mode=25 for `xgb-sem-struc-v5`
- mode=25 for `hnn-sem-struc-v5`

This module intentionally fails fast for any other feature mode.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.embed.embeddings import cosine_sim
from core.ml.lexical import (
    BM25Scorer,
    char3_jaccard,
    fuzzy_ratio,
    numeric_token_overlap,
    token_jaccard,
    token_precision,
    token_recall,
)
from core.doc.feature_extract import (
    DocumentContext,
    HeaderNode,
    classify_numbering_type,
    font_size_bucket_1pt_round,
)

RETAINED_MODE = 25

RETAINED_V5_FEATURE_NAMES: List[str] = [
    "a_b_char3_jaccard",
    "a_b_fuzz_ratio",
    "a_b_numeric_token_overlap",
    "a_b_tok_jaccard",
    "abs_font_bucket_diff",
    "abs_font_size_rank_diff",
    "abs_page_diff",
    "abs_pos_frac_diff",
    "abs_prefix_change_diff",
    "all_cap_match",
    "bm25_query_combined_b",
    "bm25_query_header_b",
    "bold_match",
    "center_match",
    "depth",
    "f1",
    "f2",
    "f3",
    "font_name_match",
    "font_size_rank_b",
    "header_text_len_b",
    "is_all_cap_b",
    "is_center_b",
    "normalized_position",
    "numbering_type_match",
    "numtype_hierarchy_score",
    "parent_structure_level_b",
    "q_b_char3_jaccard",
    "q_b_fuzz_ratio",
    "q_b_tok_jaccard",
    "q_b_tok_precision",
    "q_b_tok_recall",
    "q_path_b_tok_jaccard",
    "sim_ab_query_diff",
    "sim_b",
    "sim_query_path_b",
    "sim_query_path_diff",
    "starts_letter_match",
    "starts_num_match",
    "struc_depth",
    "struc_depth_diff",
    "struc_h1_index_norm",
    "struc_is_first_child",
    "struc_is_last_child",
    "struc_is_parent_child",
    "struc_is_same_parent",
    "struc_seq_distance",
    "struc_sibling_index_norm",
    "structure_level_diff",
    "structure_level_match",
    "textspan_len_b",
    "visual_style_match_count",
]


def _require_retained_mode(mode: int) -> None:
    if mode != RETAINED_MODE:
        raise ValueError(
            f"Artifact only supports feature mode {RETAINED_MODE}, got {mode}"
        )


def _get_query_similarities(
    header_a: HeaderNode,
    header_b: HeaderNode,
    embeddings_cache_a: Dict[str, List[float]],
    embeddings_cache_b: Dict[str, List[float]],
    query_embedding: Optional[List[float]],
    _pre_sim_a: Optional[float],
    _pre_sim_b: Optional[float],
    _pre_sim_path_a: Optional[float],
    _pre_sim_path_b: Optional[float],
) -> tuple[float, float, float, float]:
    sim_a = _pre_sim_a
    sim_b = _pre_sim_b
    sim_path_a = _pre_sim_path_a
    sim_path_b = _pre_sim_path_b

    if sim_a is None:
        a_emb = embeddings_cache_a.get(header_a.combined_text)
        sim_a = cosine_sim(a_emb, query_embedding) if a_emb and query_embedding else 0.0
    if sim_b is None:
        b_emb = embeddings_cache_b.get(header_b.combined_text)
        sim_b = cosine_sim(b_emb, query_embedding) if b_emb and query_embedding else 0.0
    if sim_path_a is None:
        path_a = header_a.processing_path
        emb_path_a = embeddings_cache_a.get(path_a) if path_a else None
        sim_path_a = (
            cosine_sim(emb_path_a, query_embedding)
            if emb_path_a and query_embedding
            else 0.0
        )
    if sim_path_b is None:
        path_b = header_b.processing_path
        emb_path_b = embeddings_cache_b.get(path_b) if path_b else None
        sim_path_b = (
            cosine_sim(emb_path_b, query_embedding)
            if emb_path_b and query_embedding
            else 0.0
        )

    return float(sim_a), float(sim_b), float(sim_path_a), float(sim_path_b)


def extract_ml_features(
    header_a: HeaderNode,
    header_b: HeaderNode,
    idx_a: int,
    idx_b: int,
    context_a: DocumentContext,
    context_b: DocumentContext,
    embeddings_cache_a: Dict[str, List[float]],
    embeddings_cache_b: Dict[str, List[float]],
    query_embedding: Optional[List[float]] = None,
    query_text: Optional[str] = None,
    mode: int = RETAINED_MODE,
    *,
    _pre_f1: Optional[float] = None,
    _pre_sim_a: Optional[float] = None,
    _pre_sim_b: Optional[float] = None,
    _pre_sim_path_a: Optional[float] = None,
    _pre_sim_path_b: Optional[float] = None,
    _pre_bm25_header_b: Optional[float] = None,
    _pre_bm25_combined_b: Optional[float] = None,
    _pre_q_b_tok_jaccard: Optional[float] = None,
    _pre_q_b_tok_recall: Optional[float] = None,
    _pre_q_b_tok_precision: Optional[float] = None,
    _pre_q_b_char3_jaccard: Optional[float] = None,
    _pre_q_b_fuzz_ratio: Optional[float] = None,
    _pre_q_path_b_tok_jaccard: Optional[float] = None,
    bm25_scorer_b: Optional[BM25Scorer] = None,
) -> Dict[str, float]:
    """Extract the retained 52-dimensional v5 feature vector."""
    _require_retained_mode(mode)

    features: Dict[str, float] = {}

    if _pre_f1 is not None:
        f1 = float(_pre_f1)
    else:
        a_emb = embeddings_cache_a.get(header_a.combined_text)
        b_emb = embeddings_cache_b.get(header_b.combined_text)
        f1 = float(cosine_sim(a_emb, b_emb) if a_emb and b_emb else 0.0)
    features["f1"] = f1

    sim_a, sim_b, sim_path_a, sim_path_b = _get_query_similarities(
        header_a,
        header_b,
        embeddings_cache_a,
        embeddings_cache_b,
        query_embedding,
        _pre_sim_a,
        _pre_sim_b,
        _pre_sim_path_a,
        _pre_sim_path_b,
    )

    features["normalized_position"] = (
        context_b.pos_frac[idx_b] if idx_b < len(context_b.pos_frac) else 0.0
    )
    features["depth"] = float(header_b.depth)

    features["f2"] = (sim_a + sim_b) / 2.0

    a_distinct = (
        context_a.prefix_pattern_approx_distinct[idx_a]
        if idx_a < len(context_a.prefix_pattern_approx_distinct)
        else 0
    )
    b_distinct = (
        context_b.prefix_pattern_approx_distinct[idx_b]
        if idx_b < len(context_b.prefix_pattern_approx_distinct)
        else 0
    )
    max_distinct = max(a_distinct, b_distinct, 1)
    features["f3"] = 1.0 - abs(a_distinct - b_distinct) / max_distinct

    features["starts_num_match"] = (
        1.0 if header_a.starts_num == header_b.starts_num else 0.0
    )
    features["starts_letter_match"] = (
        1.0 if header_a.starts_letter == header_b.starts_letter else 0.0
    )
    features["font_name_match"] = (
        1.0 if header_a.font_name == header_b.font_name else 0.0
    )
    features["bold_match"] = 1.0 if header_a.is_bold == header_b.is_bold else 0.0

    a_pos_frac = context_a.pos_frac[idx_a]
    b_pos_frac = context_b.pos_frac[idx_b]
    a_prefix_change = context_a.prefix_pattern_change_count[idx_a]
    b_prefix_change = context_b.prefix_pattern_change_count[idx_b]
    a_numtype = classify_numbering_type(header_a.text)
    b_numtype = classify_numbering_type(header_b.text)

    features["abs_page_diff"] = float(abs(header_a.page_no - header_b.page_no))
    features["abs_pos_frac_diff"] = abs(a_pos_frac - b_pos_frac)
    features["abs_font_bucket_diff"] = abs(
        font_size_bucket_1pt_round(header_a.font_size)
        - font_size_bucket_1pt_round(header_b.font_size)
    )
    features["abs_prefix_change_diff"] = float(abs(a_prefix_change - b_prefix_change))
    features["numbering_type_match"] = 1.0 if a_numtype == b_numtype else 0.0
    features["sim_ab_query_diff"] = abs(sim_a - sim_b)

    features["structure_level_diff"] = float(
        abs(header_a.structure_level - header_b.structure_level)
    )
    features["structure_level_match"] = (
        1.0 if header_a.structure_level == header_b.structure_level else 0.0
    )
    features["sim_query_path_b"] = sim_path_b
    features["sim_query_path_diff"] = abs(sim_path_a - sim_path_b)

    size_a = (
        font_size_bucket_1pt_round(header_a.font_size) if header_a.font_size else 0.0
    )
    size_b = (
        font_size_bucket_1pt_round(header_b.font_size) if header_b.font_size else 0.0
    )
    rank_a = context_a.font_size_rank_map.get(size_a, 0.5)
    rank_b = context_b.font_size_rank_map.get(size_b, 0.5)
    features["abs_font_size_rank_diff"] = abs(rank_a - rank_b)

    features["sim_b"] = sim_b
    features["struc_depth"] = float(header_b.depth)
    features["struc_h1_index_norm"] = header_b.h1_index_norm
    features["struc_sibling_index_norm"] = header_b.sibling_index_norm
    features["struc_is_first_child"] = float(header_b.is_first_child)
    features["struc_is_last_child"] = float(header_b.is_last_child)
    features["struc_depth_diff"] = float(abs(header_a.depth - header_b.depth))
    features["struc_is_same_parent"] = (
        1.0
        if header_a.parent_id == header_b.parent_id and header_a.parent_id != -1
        else 0.0
    )
    features["struc_is_parent_child"] = (
        1.0 if header_b.parent_id == header_a.idx_in_texts else 0.0
    )
    features["struc_seq_distance"] = float(
        abs(header_a.idx_in_texts - header_b.idx_in_texts)
    )

    features["all_cap_match"] = (
        1.0 if header_a.is_all_cap == header_b.is_all_cap else 0.0
    )
    features["center_match"] = (
        1.0 if header_a.is_center == header_b.is_center else 0.0
    )
    features["is_all_cap_b"] = float(header_b.is_all_cap)
    features["is_center_b"] = float(header_b.is_center)
    features["visual_style_match_count"] = float(
        sum(
            [
                header_a.is_bold == header_b.is_bold,
                header_a.is_all_cap == header_b.is_all_cap,
                header_a.is_center == header_b.is_center,
                header_a.font_name == header_b.font_name,
            ]
        )
    )
    numtype_rank = {
        "sec_item": 6,
        "roman": 5,
        "alpha": 4,
        "decimal": 3,
        "digit": 2,
        "bullet": 1,
        "none": 0,
    }
    rank_a_num = numtype_rank.get(a_numtype, 0)
    rank_b_num = numtype_rank.get(b_numtype, 0)
    features["numtype_hierarchy_score"] = (
        0.0
        if rank_a_num == rank_b_num
        else (1.0 if rank_a_num > rank_b_num else -1.0)
    )
    features["font_size_rank_b"] = context_b.font_size_rank_map.get(size_b, 0.5)

    features["header_text_len_b"] = float(len(header_b.text.split()))
    text_span = header_b.text_span or ""
    features["textspan_len_b"] = float(min(len(text_span.split()), 500))
    features["parent_structure_level_b"] = float(max(header_b.structure_level - 1, 0))

    q_text = query_text or ""
    b_text = header_b.text or ""
    b_path = header_b.processing_path or ""
    a_text = header_a.text or ""
    features["q_b_tok_jaccard"] = (
        _pre_q_b_tok_jaccard
        if _pre_q_b_tok_jaccard is not None
        else token_jaccard(q_text, b_text)
    )
    features["q_b_tok_recall"] = (
        _pre_q_b_tok_recall
        if _pre_q_b_tok_recall is not None
        else token_recall(q_text, b_text)
    )
    features["q_b_tok_precision"] = (
        _pre_q_b_tok_precision
        if _pre_q_b_tok_precision is not None
        else token_precision(q_text, b_text)
    )
    features["q_b_char3_jaccard"] = (
        _pre_q_b_char3_jaccard
        if _pre_q_b_char3_jaccard is not None
        else char3_jaccard(q_text, b_text)
    )
    features["q_b_fuzz_ratio"] = (
        _pre_q_b_fuzz_ratio
        if _pre_q_b_fuzz_ratio is not None
        else fuzzy_ratio(q_text, b_text)
    )
    features["q_path_b_tok_jaccard"] = (
        _pre_q_path_b_tok_jaccard
        if _pre_q_path_b_tok_jaccard is not None
        else token_jaccard(q_text, b_path)
    )
    features["a_b_tok_jaccard"] = token_jaccard(a_text, b_text)
    features["a_b_char3_jaccard"] = char3_jaccard(a_text, b_text)
    features["a_b_fuzz_ratio"] = fuzzy_ratio(a_text, b_text)
    features["a_b_numeric_token_overlap"] = numeric_token_overlap(a_text, b_text)

    if _pre_bm25_header_b is not None and _pre_bm25_combined_b is not None:
        features["bm25_query_header_b"] = _pre_bm25_header_b
        features["bm25_query_combined_b"] = _pre_bm25_combined_b
    elif bm25_scorer_b is not None:
        features["bm25_query_header_b"] = bm25_scorer_b.score(q_text, b_text)
        features["bm25_query_combined_b"] = bm25_scorer_b.score(
            q_text, header_b.combined_text or ""
        )
    else:
        features["bm25_query_header_b"] = 0.0
        features["bm25_query_combined_b"] = 0.0

    return features


__all__ = [
    "RETAINED_MODE",
    "RETAINED_V5_FEATURE_NAMES",
    "extract_ml_features",
]
