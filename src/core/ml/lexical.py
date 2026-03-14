# -*- coding: utf-8 -*-
"""
Lexical feature utilities.

Domain-agnostic text normalization and overlap/matching metrics used by mode=22.
"""

from __future__ import annotations

from collections import Counter
import difflib
import math
import re
import unicodedata
from functools import lru_cache
from typing import FrozenSet, Sequence, Tuple

import numpy as np

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover - exercised in environments without rapidfuzz
    _rf_fuzz = None


_RE_WORD = re.compile(r"[a-z0-9]+")
_RE_NUMERIC = re.compile(r"\d+(?:[.,]\d+)?")


@lru_cache(maxsize=500_000)
def _normalize_text_cached(text: str) -> str:
    norm = unicodedata.normalize("NFKC", text).lower()
    return " ".join(norm.split())


def normalize_text(text: str) -> str:
    """Normalize text for lexical comparisons."""
    if not text:
        return ""
    return _normalize_text_cached(str(text))


@lru_cache(maxsize=500_000)
def _token_set(text: str) -> FrozenSet[str]:
    return frozenset(_RE_WORD.findall(normalize_text(text)))


@lru_cache(maxsize=500_000)
def _char3_set(text: str) -> FrozenSet[str]:
    normalized = normalize_text(text).replace(" ", "")
    if not normalized:
        return frozenset()
    if len(normalized) < 3:
        return frozenset({normalized})
    return frozenset(normalized[i : i + 3] for i in range(len(normalized) - 2))


@lru_cache(maxsize=500_000)
def _numeric_set(text: str) -> FrozenSet[str]:
    return frozenset(_RE_NUMERIC.findall(normalize_text(text)))


def _jaccard(set_a: FrozenSet[str], set_b: FrozenSet[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b)) / float(len(union))


def token_jaccard(text_a: str, text_b: str) -> float:
    return _jaccard(_token_set(text_a), _token_set(text_b))


def token_recall(query_text: str, candidate_text: str) -> float:
    query_tokens = _token_set(query_text)
    if not query_tokens:
        return 0.0
    cand_tokens = _token_set(candidate_text)
    return float(len(query_tokens & cand_tokens)) / float(len(query_tokens))


def token_precision(query_text: str, candidate_text: str) -> float:
    query_tokens = _token_set(query_text)
    cand_tokens = _token_set(candidate_text)
    if not cand_tokens:
        return 0.0
    return float(len(query_tokens & cand_tokens)) / float(len(cand_tokens))


def char3_jaccard(text_a: str, text_b: str) -> float:
    return _jaccard(_char3_set(text_a), _char3_set(text_b))


def fuzzy_ratio(text_a: str, text_b: str) -> float:
    """Return normalized [0,1] fuzzy matching score."""
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0

    if _rf_fuzz is not None:
        return float(_rf_fuzz.ratio(norm_a, norm_b)) / 100.0

    # Fallback when rapidfuzz is unavailable.
    return float(difflib.SequenceMatcher(a=norm_a, b=norm_b).ratio())


def numeric_token_overlap(text_a: str, text_b: str) -> float:
    nums_a = _numeric_set(text_a)
    nums_b = _numeric_set(text_b)
    if not nums_a and not nums_b:
        return 0.0
    denom = max(len(nums_a), len(nums_b))
    if denom == 0:
        return 0.0
    return float(len(nums_a & nums_b)) / float(denom)


# ========== BM25 ==========


@lru_cache(maxsize=500_000)
def _token_list(text: str) -> Tuple[str, ...]:
    """Return token list (preserving duplicates, used for TF computation)."""
    return tuple(_RE_WORD.findall(normalize_text(text)))


class BM25Scorer:
    """
    Corpus-based BM25 scorer.

    Usage:
        scorer = BM25Scorer(corpus_texts)  # initialize IDF from document corpus
        score = scorer.score(query, doc)
    """

    def __init__(self, corpus_texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Build IDF
        doc_count = len(corpus_texts)
        df = {}  # document frequency
        total_len = 0.0
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lens = np.zeros(doc_count, dtype=np.float32)

        for i, text in enumerate(corpus_texts):
            token_list = _token_list(text)
            token_set = set(token_list)
            tf_map: Counter[str] = Counter(token_list)
            doc_len = float(len(token_list))
            self._doc_term_freqs.append(tf_map)
            self._doc_lens[i] = doc_len
            total_len += doc_len
            for t in token_set:
                df[t] = df.get(t, 0) + 1

        self.avgdl = total_len / max(float(doc_count), 1.0)
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1) -- standard BM25 formula
        self.idf = {
            term: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def score(self, query: str, doc: str) -> float:
        """Compute the BM25 score of a query against a document."""
        q_tokens = tuple(dict.fromkeys(_token_list(query)))
        d_tokens = _token_list(doc)
        if not q_tokens or not d_tokens:
            return 0.0

        dl = float(len(d_tokens))
        tf_map: Counter[str] = Counter(d_tokens)
        k_factor = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))

        score = 0.0
        for qt in q_tokens:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            idf = self.idf.get(qt, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + k_factor
            score += idf * numerator / denominator

        return score

    def score_batch(self, query: str, docs: Sequence[str]) -> "np.ndarray":
        """Batch-compute BM25 scores, avoiding redundant tokenization and TF counting."""
        n_docs = len(docs)
        if n_docs == 0:
            return np.zeros(0, dtype=np.float32)

        q_terms = tuple(dict.fromkeys(_token_list(query)))
        if not q_terms:
            return np.zeros(n_docs, dtype=np.float32)

        term_idf = {
            term: idf for term in q_terms if (idf := self.idf.get(term, 0.0)) > 0.0
        }
        if not term_idf:
            return np.zeros(n_docs, dtype=np.float32)

        tokenized_docs = [_token_list(doc) for doc in docs]
        doc_lens = np.fromiter(
            (len(tokens) for tokens in tokenized_docs),
            dtype=np.float32,
            count=n_docs,
        )
        tf_maps = [Counter(tokens) for tokens in tokenized_docs]
        k_factors = self.k1 * (1.0 - self.b + self.b * doc_lens / max(self.avgdl, 1e-9))
        scores = np.zeros(n_docs, dtype=np.float32)

        for i, (tf_map, k_factor) in enumerate(zip(tf_maps, k_factors)):
            score = 0.0
            for term, idf in term_idf.items():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + k_factor
                score += idf * numerator / denominator
            scores[i] = score

        return scores


# ========== Batch Functions for Performance ==========


def batch_token_jaccard(query: str, texts: Sequence[str]) -> np.ndarray:
    """Batch-compute token Jaccard between a query and multiple texts, reducing call overhead."""
    query_tokens = _token_set(query)
    return np.array(
        [_jaccard(query_tokens, _token_set(t)) for t in texts], dtype=np.float32
    )


def batch_token_recall(query: str, texts: Sequence[str]) -> np.ndarray:
    """Batch-compute token recall."""
    query_tokens = _token_set(query)
    if not query_tokens:
        return np.zeros(len(texts), dtype=np.float32)
    return np.array(
        [len(query_tokens & _token_set(t)) / len(query_tokens) for t in texts],
        dtype=np.float32,
    )


def batch_token_precision(query: str, texts: Sequence[str]) -> np.ndarray:
    """Batch-compute token precision."""
    query_tokens = _token_set(query)
    cand_token_sets = [_token_set(t) for t in texts]
    return np.array(
        [
            (len(query_tokens & cand) / len(cand)) if cand else 0.0
            for cand in cand_token_sets
        ],
        dtype=np.float32,
    )


def batch_char3_jaccard(query: str, texts: Sequence[str]) -> np.ndarray:
    """Batch-compute character 3-gram Jaccard."""
    query_chars = _char3_set(query)
    return np.array(
        [_jaccard(query_chars, _char3_set(t)) for t in texts], dtype=np.float32
    )


def batch_fuzzy_ratio(query: str, texts: Sequence[str]) -> np.ndarray:
    """Batch-compute fuzzy ratio (uses rapidfuzz process.cdist when available)."""
    if _rf_fuzz is None:
        # fallback to per-item computation
        return np.array([fuzzy_ratio(query, t) for t in texts], dtype=np.float32)

    from rapidfuzz import process

    norm_query = normalize_text(query)
    norm_texts = [normalize_text(t) for t in texts]

    # process.cdist is a high-performance C-level batch computation
    scores = process.cdist(
        [norm_query], norm_texts, scorer=_rf_fuzz.ratio, score_cutoff=0.0
    )[0]  # take the first row
    return (scores / 100.0).astype(np.float32)


# ========== Cache Monitoring ==========


def get_lexical_cache_info():
    """Return lexical cache hit-rate info for performance debugging."""
    return {
        "normalize_text": _normalize_text_cached.cache_info(),
        "token_set": _token_set.cache_info(),
        "char3_set": _char3_set.cache_info(),
        "numeric_set": _numeric_set.cache_info(),
        "token_list": _token_list.cache_info(),
    }


def clear_lexical_cache():
    """Clear all lexical caches (call when memory is tight)."""
    _normalize_text_cached.cache_clear()
    _token_set.cache_clear()
    _char3_set.cache_clear()
    _numeric_set.cache_clear()
    _token_list.cache_clear()
