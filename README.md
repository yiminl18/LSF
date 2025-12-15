### core/feature_extract.py — README 

#### Purpose

core/feature_extract.py extracts structured features from Docling-style *_merged.json files.

It supports two use-cases:

- Node-level features: describe a single header node (form/style/structure).

- Pair-level features: describe a pair of nodes (a, b) and are used as input to an XGBoost classifier that learns Similarity(a, b).

Design goals: clear, extensible, reusable, and robust to OCR noise (e.g., font-size bucketing).

------

#### Input format (from *_merged.json)

The document is expected to contain a texts array (merged["texts"]) where each item is a node dict. Common fields used:

- content_layer: "body" / "furniture"

- label: e.g. "section_header", "page_header"

- text: header string

- text_span: extra text associated with the header (may be empty)

- prov[0].page_no: page number

- size: font size (float)

- bold: bold flag (0/1)

------

#### Header candidate filtering (current behavior)

Only keep nodes satisfying:

- content_layer == "body"

- label == "section_header"

The function preserves the original texts order (no re-sorting).

Entry point:

- iter_section_headers(merged: dict) -> List[HeaderNode]

------

#### Core data structures

##### HeaderNode

A lightweight wrapper of a merged node:

- idx_in_texts: original index in merged["texts"]

- text: node text

- text_span: node text span

- page_no: integer page number (0 if missing)

- font_size: float font size (0.0 if missing)

- is_bold: 0/1

Convenience property:

- combined_text = f"{text} {text_span}".strip()

This combined_text is intended to align with the embedding cache keys used elsewhere in the repo (document embeddings stored as {combined_text: embedding_vector}).

------

#### Numbering type (numbering_type)

A coarse, prefix-based classification for header text. Used for structural cues and pattern counting.

Supported classes:

- none

- digit (e.g., 1, 2., (3))

- decimal (e.g., 1.2, 2.3.4)

- alpha (e.g., A., (b))

- roman (e.g., I., (iv))

- bullet (e.g., •, -, *)

- sec_item (SEC-style headings, includes ITEM and PART)

Priority order (first match wins):

1. bullet

1. sec_item

1. decimal

1. digit

1. alpha

1. roman

1. none

API:

- classify_numbering_type(text: str) -> str

------

#### Font-size bucketing (1pt, round)

To reduce OCR jitter, font sizes are bucketed to the nearest 1pt:

- font_size_bucket = round(font_size / 1.0) * 1.0

API:

- font_size_bucket_1pt_round(font_size: float) -> float

------

#### Document-level context (DocumentContext)

Prefix features require a single pass over the header list.

API:

- build_document_context(headers_in_order: Sequence[HeaderNode]) -> DocumentContext

Outputs:

- total_headers: number of header nodes

- pos_frac[i]: i / total_headers (0..1)

- pattern_key[i]: (font_size_bucket_1pt_round, is_bold, numbering_type)

- prefix_pattern_change_count[i]: number of pattern_key changes in [0, i)

- prefix_pattern_approx_distinct[i]: number of distinct pattern_key values in [0, i)

Note: prefix statistics exclude the node itself (computed from the prefix [0, i)).

------

#### Node-level feature extraction

API:

- extract_node_features(headers_in_order: Sequence[HeaderNode], ctx: DocumentContext) -> List[Dict[str, float]]

For each header node i, the output dict contains:

- starts_with_number (0/1)

- starts_with_letter (0/1)

- font_size_bucket_1pt_round (float)

- is_bold (0/1)

- page_no (float)

- pos_frac (float)

- prefix_pattern_change_count (float)

- prefix_pattern_approx_distinct (float)

- One-hot numbering type:

- numtype_none, numtype_digit, numtype_decimal, numtype_alpha,

numtype_roman, numtype_bullet, numtype_sec_item

Important: node-level features do not include rank_index or sim(q,node) by design.

------

#### Pair-level feature extraction (Similarity model input)

API:

- extract_pair_features(a: HeaderNode, b: HeaderNode, a_feats: dict, b_feats: dict, sim_node_node: float) -> Dict[str, float]

Where:

- sim_node_node is expected to be embedding cosine similarity between a.combined_text and b.combined_text.

Output features:

- sim_node_node

- abs_font_bucket_diff

- bold_match (0/1)

- numbering_type_match (0/1)

- abs_page_diff

- abs_pos_frac_diff

- abs_prefix_change_diff

- abs_prefix_distinct_diff

These are designed to feed an XGBClassifier that predicts P(label=1 | a,b).

------

#### Typical usage pattern

1) Load merged JSON.
2) Extract headers:

- headers = iter_section_headers(merged)
- Build doc context:

- ctx = build_document_context(headers)
- Compute node features:

- node_feats = extract_node_features(headers, ctx)
- For a pair (i, j) compute:

- sim_node_node = cosine(emb(headers[i].combined_text), emb(headers[j].combined_text))

- pair_feats = extract_pair_features(headers[i], headers[j], node_feats[i], node_feats[j], sim_node_node)

------

#### Notes / current assumptions

- Candidate set is restricted to body + section_header. If you later want cover-page fields or other labels, update iter_section_headers.

- Prefix features assume the texts order is meaningful reading/structural order (we intentionally do not re-sort here).

- Embeddings are assumed to be keyed by combined_text (text + text_span), matching existing embedding cache behavior in this repo.