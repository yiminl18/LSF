"""GraphRAG baseline: entity graph + community detection + community summary retrieval.

Algorithm:
1. Flatten document and split into chunks.
2. Extract entities and relations from each chunk (LLM).
3. Build an undirected graph (nodes=entities, edges=relations).
4. Louvain community detection.
5. Generate a summary for each community (LLM).
6. Embed community summaries; rank by cosine similarity at retrieval time.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("GraphRAG baseline requires networkx; install with: pip install networkx") from e

from core.embed.embeddings import cosine_sim_batch, get_embeddings_batch, get_model_name_for_provider
from core.pipeline.e2e_utils.baselines import register_baseline
from core.pipeline.e2e_utils.baselines.base import BaseRAGBaseline
from core.pipeline.e2e_utils.rag_vanilla import chunk_text, flatten_document

# Entity/relation extraction prompt
_ENTITY_EXTRACT_PROMPT = """\
Extract entities and relationships from the following text.
Return as JSON with two arrays:
- "entities": [{{"name": "...", "type": "...", "description": "..."}}]
- "relationships": [{{"source": "...", "target": "...", "description": "..."}}]
Only extract clear, factual items. If none found, return {{"entities": [], "relationships": []}}.

Text: {chunk_text}"""

# Community summary prompt
_COMMUNITY_SUMMARY_PROMPT = """\
Summarize the following community of related entities and their relationships.
Focus on the key facts and connections.

Entities and relationships:
{community_context}

Summary:"""


@register_baseline("rag-graph")
class GraphRAGBaseline(BaseRAGBaseline):
    """GraphRAG baseline: entity graph + Louvain community detection + community summary vector retrieval."""

    name = "rag-graph"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Memory cache: {doc_id: (community_summaries, summary_embeddings)}
        self._mem_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    def _cache_dir(self, doc_id: str) -> Path:
        """Return the graphrag cache directory for this document."""
        return self.cache_root / "rag_graphrag" / doc_id

    def _is_cache_complete(self, doc_id: str) -> bool:
        """Check whether the disk cache is complete (all four files present)."""
        d = self._cache_dir(doc_id)
        return all(
            (d / fname).exists()
            for fname in ("graph.json", "communities.json", "summaries.json", "summary_embeddings.npz")
        )

    def _extract_entities_from_chunk(self, chunk_text_str: str, chunk_idx: int) -> tuple[list[dict], list[dict]]:
        """Call the LLM to extract entities and relations from a single chunk; returns (entities, relationships)."""
        prompt = _ENTITY_EXTRACT_PROMPT.format(chunk_text=chunk_text_str)
        result = self.cached_caller.call(
            prompt,
            llm_provider=self.llm_provider,
            model=self.llm_model,
            max_tokens=800,
        )
        self.track_preprocess_call(result)
        response = result.response

        # Fault-tolerant JSON parsing.
        try:
            # Try direct parse first.
            data = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON substring.
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(response[start:end])
                else:
                    return [], []
            except (json.JSONDecodeError, ValueError):
                return [], []

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Record the source chunk index for each relation.
        for rel in relationships:
            rel["source_chunk_idx"] = chunk_idx

        return entities, relationships

    def _build_graph(
        self,
        all_entities: list[dict],
        all_relationships: list[dict],
    ) -> nx.Graph:
        """Build an undirected graph from entities and relations.

        Entities with the same name have their descriptions merged; edge attributes
        include description and source_chunk_idx.
        """
        G = nx.Graph()

        # Add/merge nodes (normalize: lowercase + strip).
        for ent in all_entities:
            name = ent.get("name", "").strip().lower()
            if not name:
                continue
            desc = ent.get("description", "").strip()
            ent_type = ent.get("type", "").strip()

            if G.has_node(name):
                # Merge description (deduplicated concatenation).
                existing_desc = G.nodes[name].get("description", "")
                if desc and desc not in existing_desc:
                    G.nodes[name]["description"] = f"{existing_desc}; {desc}".strip("; ")
            else:
                G.add_node(name, type=ent_type, description=desc)

        # Add edges.
        for rel in all_relationships:
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
            if not src or not tgt or src == tgt:
                continue
            # Ensure nodes exist.
            if not G.has_node(src):
                G.add_node(src, type="", description="")
            if not G.has_node(tgt):
                G.add_node(tgt, type="", description="")
            G.add_edge(
                src, tgt,
                description=rel.get("description", ""),
                source_chunk_idx=rel.get("source_chunk_idx", -1),
            )

        return G

    def _detect_communities(self, G: nx.Graph) -> list[set[str]]:
        """Louvain community detection. When the graph is too small, each node forms its own community."""
        if G.number_of_nodes() < 3:
            return [{node} for node in G.nodes()]

        try:
            communities = list(nx.community.louvain_communities(G, seed=42))
        except (ValueError, RuntimeError) as e:
            print(f"  Louvain failed ({type(e).__name__}), falling back to connected components",
                  file=sys.stderr)
            communities = list(nx.connected_components(G))

        return communities

    def _build_community_context(
        self,
        community: set[str],
        G: nx.Graph,
        chunks: list[str],
    ) -> str:
        """Build context text for community summarization.

        Includes: entity descriptions within the community + related edge descriptions
        + source chunk text (deduplicated).
        """
        parts: list[str] = []

        # Entity descriptions.
        for node in sorted(community):
            if not G.has_node(node):
                continue
            attrs = G.nodes[node]
            desc = attrs.get("description", "")
            ent_type = attrs.get("type", "")
            if desc:
                parts.append(f"Entity: {node} ({ent_type}): {desc}")

        # Related edges (intra-community or cross-community).
        seen_chunk_indices: set[int] = set()
        for u, v, data in G.edges(data=True):
            if u not in community and v not in community:
                continue
            edge_desc = data.get("description", "")
            if edge_desc:
                parts.append(f"Relation: {u} -> {v}: {edge_desc}")
            chunk_idx = data.get("source_chunk_idx", -1)
            if chunk_idx >= 0:
                seen_chunk_indices.add(chunk_idx)

        # Source chunk text (truncated to avoid excessive length).
        for idx in sorted(seen_chunk_indices):
            if idx < len(chunks):
                parts.append(f"Source text: {chunks[idx][:300]}")

        return "\n".join(parts)

    def _generate_community_summary(self, community_context: str) -> str:
        """Call the LLM to generate a summary for a single community."""
        prompt = _COMMUNITY_SUMMARY_PROMPT.format(community_context=community_context)
        result = self.cached_caller.call(
            prompt,
            llm_provider=self.llm_provider,
            model=self.llm_model,
            max_tokens=400,
        )
        self.track_preprocess_call(result)
        return result.response

    def preprocess_doc(self, doc_id: str) -> None:
        """Preprocess a single document: build entity graph + community detection + generate summaries, then cache."""
        # Skip if cache is complete.
        if self._is_cache_complete(doc_id):
            return

        json_path = self.processing_dir / f"{doc_id}_reconstructed.json"
        if not json_path.exists():
            print(f"  WARNING: {json_path} not found, skipping", file=sys.stderr)
            return

        # Flatten and split.
        flat_text = flatten_document(json_path)
        chunks = chunk_text(flat_text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return

        # Extract entities and relations from all chunks.
        all_entities: list[dict] = []
        all_relationships: list[dict] = []
        for idx, chunk_str in enumerate(chunks):
            entities, relationships = self._extract_entities_from_chunk(chunk_str, idx)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Build graph.
        G = self._build_graph(all_entities, all_relationships)

        # Fall back to empty communities when graph is empty.
        if G.number_of_nodes() == 0:
            communities: list[set[str]] = []
        else:
            communities = self._detect_communities(G)

        # Generate a summary for each community.
        community_data: list[dict] = []
        summaries: list[str] = []
        for community in communities:
            context = self._build_community_context(community, G, chunks)
            if not context.strip():
                summary = " ".join(sorted(community))
            else:
                summary = self._generate_community_summary(context)
            community_data.append({"entity_names": sorted(community), "summary": summary})
            summaries.append(summary)

        # Batch-embed community summaries.
        if summaries:
            model_name = get_model_name_for_provider(self.embed_provider)
            summary_embs = np.array(
                get_embeddings_batch(summaries, model=model_name, provider=self.embed_provider),
                dtype=np.float32,
            )
        else:
            summary_embs = np.zeros((0, 1), dtype=np.float32)

        # Serialize graph data.
        graph_data = {
            "nodes": [
                {"name": n, "type": G.nodes[n].get("type", ""), "desc": G.nodes[n].get("description", "")}
                for n in G.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "desc": d.get("description", ""),
                    "chunk_idx": d.get("source_chunk_idx", -1),
                }
                for u, v, d in G.edges(data=True)
            ],
        }

        # Write to disk cache.
        cache_dir = self._cache_dir(doc_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False)

        with open(cache_dir / "communities.json", "w", encoding="utf-8") as f:
            json.dump(community_data, f, ensure_ascii=False)

        with open(cache_dir / "summaries.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False)

        np.savez_compressed(cache_dir / "summary_embeddings.npz", embeddings=summary_embs)

    def _load_cache(self, doc_id: str) -> tuple[list[str], np.ndarray] | None:
        """Load community summaries + embeddings from memory or disk cache."""
        if doc_id in self._mem_cache:
            return self._mem_cache[doc_id]

        cache_dir = self._cache_dir(doc_id)
        summaries_path = cache_dir / "summaries.json"
        emb_path = cache_dir / "summary_embeddings.npz"

        if not summaries_path.exists() or not emb_path.exists():
            return None

        with open(summaries_path, "r", encoding="utf-8") as f:
            summaries: list[str] = json.load(f)

        data = np.load(emb_path)
        embs: np.ndarray = data["embeddings"]

        self._mem_cache[doc_id] = (summaries, embs)
        return summaries, embs

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        doc_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k community summaries by cosine similarity."""
        cached = self._load_cache(doc_id)
        if cached is None:
            return []

        summaries, embs = cached
        if embs.shape[0] == 0:
            return []

        scores = cosine_sim_batch(embs, np.array(query_embedding, dtype=np.float32))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(summaries[i], float(scores[i])) for i in top_indices]
