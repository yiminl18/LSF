"""HippoRAG baseline: knowledge graph + Personalized PageRank retrieval.

Preprocessing builds a knowledge graph via spaCy NER + LLM triple extraction;
retrieval performs Personalized PageRank on the graph using query entities,
falling back to cosine retrieval when no entities match.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import spacy
except ImportError as e:
    raise ImportError("HippoRAG requires spacy: pip install spacy && python -m spacy download en_core_web_sm") from e

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("HippoRAG requires networkx: pip install networkx") from e

from core.embed.embeddings import cosine_sim_batch
from core.pipeline.e2e_utils.baselines import register_baseline
from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline
from core.pipeline.e2e_utils.cache import CachedLLMCaller
from core.pipeline.e2e_utils.rag_vanilla import (
    chunk_text,
    embed_chunks,
    flatten_document,
    retrieve_chunks,
)

# NER entity types to retain
_NER_LABELS = {
    "PERSON", "ORG", "GPE", "DATE", "MONEY",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
}

# Triple extraction prompt
_TRIPLE_PROMPT = """\
Extract knowledge triples (subject, relation, object) from the following text.
Return as JSON array of objects with keys "s", "r", "o".
Only extract factual relationships. If none found, return [].

Text: {chunk_text}

Triples:"""


def _extract_entities(nlp, text: str) -> list[str]:
    """Extract and normalize entities with spaCy (lowercase + strip)."""
    doc = nlp(text)
    return list({
        ent.text.lower().strip()
        for ent in doc.ents
        if ent.label_ in _NER_LABELS and ent.text.strip()
    })


def _extract_triples(
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    chunk_text_str: str,
    baseline: "BaseRAGBaseline | None" = None,
) -> list[dict]:
    """Use the LLM to extract (s, r, o) triples from a chunk; returns empty list on parse failure."""
    prompt = _TRIPLE_PROMPT.format(chunk_text=chunk_text_str)
    result = cached_caller.call(
        prompt,
        llm_provider=llm_provider,
        model=llm_model,
        max_tokens=500,
    )
    if baseline is not None:
        baseline.track_preprocess_call(result)
    try:
        # Extract the JSON array from the response.
        text = result.response.strip()
        # Match from the first '[' to the last ']'.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        triples = json.loads(match.group())
        if not isinstance(triples, list):
            return []
        # Filter well-formed triples.
        valid = []
        for t in triples:
            if isinstance(t, dict) and all(k in t for k in ("s", "r", "o")):
                valid.append({
                    "s": str(t["s"]).lower().strip(),
                    "r": str(t["r"]).lower().strip(),
                    "o": str(t["o"]).lower().strip(),
                })
        return valid
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  WARNING: triple extraction failed: {type(e).__name__}", file=sys.stderr)
        return []


@register_baseline("rag-hippo")
class HippoRAGBaseline(BaseRAGBaseline):
    """HippoRAG: knowledge graph construction + Personalized PageRank retrieval."""

    name = "rag-hippo"

    def __init__(
        self,
        processing_dir: Path,
        embed_provider: str,
        cached_caller: CachedLLMCaller,
        llm_provider: str,
        llm_model: str,
        cache_root: Path = Path(".cache"),
        chunk_size: int = 200,
        chunk_overlap: int = 0,
        ner_model: str = "en_core_web_sm",
    ) -> None:
        super().__init__(
            processing_dir=processing_dir,
            embed_provider=embed_provider,
            cached_caller=cached_caller,
            llm_provider=llm_provider,
            llm_model=llm_model,
            cache_root=cache_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Load spaCy model once.
        self._nlp = spacy.load(ner_model)
        self._hippo_cache_dir = cache_root / "rag_hippo"
        self._hippo_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, doc_id: str) -> Path:
        return self._hippo_cache_dir / f"{doc_id}.json"

    def preprocess_doc(self, doc_id: str) -> None:
        """Build knowledge graph and cache to disk; skips if cache already exists."""
        cache_path = self._cache_path(doc_id)
        if cache_path.exists():
            return

        # 1. Flatten document and split into chunks.
        json_path = self.processing_dir / f"{doc_id}_reconstructed.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping", file=sys.stderr)
            return

        full_text = flatten_document(json_path)
        chunks = chunk_text(full_text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return

        # 2. Extract entities and triples from each chunk.
        all_edges: list[dict] = []
        entity_to_chunks: dict[str, list[int]] = {}
        chunk_to_entities: dict[str, list[str]] = {}

        for idx, chunk in enumerate(chunks):
            # NER entity extraction.
            entities = _extract_entities(self._nlp, chunk)
            # Triple extraction (via LLM, with caching).
            triples = _extract_triples(
                self.cached_caller,
                self.llm_provider,
                self.llm_model,
                chunk,
                baseline=self,
            )

            # Include triple nodes in the entity set.
            triple_entities: set[str] = set()
            for t in triples:
                triple_entities.add(t["s"])
                triple_entities.add(t["o"])
                all_edges.append({**t, "chunk_idx": idx})

            all_entities = list(set(entities) | triple_entities)
            chunk_to_entities[str(idx)] = all_entities
            for ent in all_entities:
                entity_to_chunks.setdefault(ent, []).append(idx)

        # 3. Collect all unique nodes.
        nodes = list({
            e
            for ents in chunk_to_entities.values()
            for e in ents
        })

        # 4. Write to disk cache.
        cache_data = {
            "chunks": chunks,
            "nodes": nodes,
            "edges": all_edges,
            "entity_to_chunks": entity_to_chunks,
            "chunk_to_entities": chunk_to_entities,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def _load_graph_cache(self, doc_id: str) -> dict | None:
        """Load knowledge graph cache from disk; returns None if not found."""
        cache_path = self._cache_path(doc_id)
        if not cache_path.exists():
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Personalized PageRank retrieval; falls back to cosine retrieval when no entities match."""
        cache = self._load_graph_cache(doc_id)
        if cache is None:
            return []

        chunks: list[str] = cache["chunks"]
        nodes: list[str] = cache["nodes"]
        edges: list[dict] = cache["edges"]
        entity_to_chunks: dict[str, list[int]] = cache["entity_to_chunks"]

        if not chunks:
            return []

        # 1. Query NER.
        query_entities = _extract_entities(self._nlp, query)

        # 2. Find matching nodes in the graph (exact + substring).
        node_set = set(nodes)
        matched_nodes: set[str] = set()
        for qe in query_entities:
            if qe in node_set:
                matched_nodes.add(qe)
            else:
                # Substring match.
                for n in nodes:
                    if qe in n or n in qe:
                        matched_nodes.add(n)

        # 3. No matches — fall back to cosine retrieval.
        if not matched_nodes:
            return self._cosine_fallback(query_embedding, chunks, doc_id, top_k)

        # 4. Build DiGraph.
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for edge in edges:
            s, o = edge["s"], edge["o"]
            if s in node_set and o in node_set:
                G.add_edge(s, o, relation=edge["r"])

        # 5. Personalized PageRank.
        weight = 1.0 / len(matched_nodes)
        personalization = {n: weight for n in matched_nodes}
        try:
            pr_scores: dict[str, float] = nx.pagerank(
                G, personalization=personalization, alpha=0.85
            )
        except nx.PowerIterationFailedConvergence:
            return self._cosine_fallback(query_embedding, chunks, doc_id, top_k)

        # 6. Rank chunks by the highest PageRank score of their associated entities.
        chunk_scores: dict[int, float] = {}
        for ent, chunk_indices in entity_to_chunks.items():
            ent_score = pr_scores.get(ent, 0.0)
            for ci in chunk_indices:
                if ci < len(chunks):
                    chunk_scores[ci] = max(chunk_scores.get(ci, 0.0), ent_score)

        if not chunk_scores:
            return self._cosine_fallback(query_embedding, chunks, doc_id, top_k)

        # Sort descending by score and take top_k.
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunks[ci], score) for ci, score in sorted_chunks[:top_k]]

    def _cosine_fallback(
        self,
        query_embedding: list[float],
        chunks: list[str],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Fall back to cosine similarity retrieval when entity matching fails."""
        chunk_embs: np.ndarray = embed_chunks(
            chunks,
            self.embed_provider,
            doc_id,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        return retrieve_chunks(query_embedding, chunk_embs, chunks, top_k=top_k)
