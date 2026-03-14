"""Cross-Encoder semantic verification module.

Uses the NLI three-class output from nli-deberta-v3-small (~280MB) for semantic verification:
1. Header validity verification (NLI framing, replacing GPT-4o calls)
2. Parent-child semantic relationship correction (direct entailment scoring)

The NLI model outputs three logits: [contradiction, neutral, entailment].
- Higher entailment score indicates stronger semantic relevance.
- Empirical parent-child discrimination: correct pairs ~+4, incorrect pairs ~-2 (gap ~6 points).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.doc.tree_reconstructor import Node

# Module-level singleton cache (same pattern as _local_model_cache in embeddings.py)
_ce_cache: Dict[str, object] = {}

CE_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_MAX_TEXT_LEN = 500
# NLI three-class output indices
_IDX_ENTAILMENT = 2
_IDX_CONTRADICTION = 0
# Optimized batch size for MPS (32 is stable for DeBERTa-V3 on Mac)
DEFAULT_BATCH_SIZE = 32


def _load_cross_encoder(model_name: str = CE_MODEL_NAME):
    """Lazily load the Cross-Encoder model (singleton)."""
    global _ce_cache
    if model_name not in _ce_cache:
        from sentence_transformers import CrossEncoder

        print(f"Loading cross-encoder model: {model_name}...")
        _ce_cache[model_name] = CrossEncoder(model_name)
    return _ce_cache[model_name]


def _predict_entailment(
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Batch inference to extract entailment scores.

    Args:
        pairs: list of (text_a, text_b) tuples
        batch_size: model inference batch size

    Returns:
        float32 entailment score array of shape (N,)
    """
    if not pairs:
        return np.array([], dtype=np.float32)

    model = _load_cross_encoder()
    truncated = [(a[:_MAX_TEXT_LEN], b[:_MAX_TEXT_LEN]) for a, b in pairs]
    raw = model.predict(truncated, batch_size=batch_size)
    logits = np.asarray(raw, dtype=np.float32)
    # NLI model returns (N, 3); extract the entailment column
    if logits.ndim == 2 and logits.shape[1] >= 3:
        return logits[:, _IDX_ENTAILMENT]
    # Non-NLI model fallback
    return logits


def score_header_content_pairs(
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Batch-score entailment for (text_a, text_b) pairs.

    Accepts raw pairs directly; suitable for parent-child relationship verification.

    Args:
        pairs: list of (parent_text, child_text) tuples
        batch_size: model inference batch size

    Returns:
        float32 score array of shape (N,); higher values indicate stronger semantic relevance
    """
    return _predict_entailment(pairs, batch_size)


def score_header_validity(
    header_context_pairs: Sequence[Tuple[str, str]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Evaluate whether header text is a valid section heading for the following content.

    Uses NLI framing: premise=content, hypothesis="This section is about {header}."
    This lets the NLI model judge whether the content entails the section heading,
    providing better discrimination than direct comparison.

    Args:
        header_context_pairs: list of (header_text, context_text) tuples
        batch_size: model inference batch size

    Returns:
        float32 entailment score array of shape (N,)
    """
    if not header_context_pairs:
        return np.array([], dtype=np.float32)

    # NLI framing: premise=context, hypothesis="This section is about {header}."
    nli_pairs = [
        (ctx[:_MAX_TEXT_LEN], f"This section is about {header}.")
        for header, ctx in header_context_pairs
    ]
    return _predict_entailment(nli_pairs, batch_size)


def _parent_quality(node: "Node") -> float:
    """Evaluate the quality of a parent node as a header (0~1).

    Low-quality parents (non-bold, digit-prefixed, overly long text) require
    higher NLI scores to retain the parent-child relationship.
    """
    from core.doc.feature_extract import classify_numbering_type

    q = 1.0
    if not node.style.get("bold"):
        q -= 0.3
    if classify_numbering_type(node.text) == "digit":
        q -= 0.3
    if len(node.text) > 120:
        q -= 0.2
    return max(0.0, q)


def _collect_ancestor_chain(node: "Node") -> List["Node"]:
    """Collect the ancestor chain of a node (excluding root)."""
    ancestors = []
    cur = node.parent
    while cur is not None and cur.label != "root":
        ancestors.append(cur)
        cur = cur.parent
    return ancestors


def collect_parent_verification_tasks(
    root: "Node",
    body_style: Dict[str, Any],
) -> List[Tuple["Node", "Node"]]:
    """DFS to collect (parent, child) pairs that need verification.

    Special handling: if parent is a text node and child is a section_header, force collection for repair.
    """
    from core.doc.feature_extract import classify_numbering_type

    body_size = body_style["size"]
    pairs_to_check: List[Tuple["Node", "Node"]] = []

    def _collect(node: "Node") -> None:
        for child in node.children:
            _collect(child)
        if not node.children:
            return
        for child in node.children:
            # Special case: text node as parent, section_header as child — force collection
            if node.label == "text" and child.label == "section_header":
                pairs_to_check.append((node, child))
                continue

            if not child.children and child.label != "section_header":
                continue
            # Short-circuit
            numtype = classify_numbering_type(child.text)
            if numtype == "sec_item" or child.style.get("size", 0) > body_size * 1.3:
                continue
            if node.label == "root":
                continue
            pairs_to_check.append((node, child))

    _collect(root)
    return pairs_to_check


def apply_parent_fixes(
    pairs_to_check: List[Tuple["Node", "Node"]],
    scores: np.ndarray,
    threshold: float = 0.0,
    quality_penalty: float = 1.5,
) -> int:
    """Apply re-parenting based on inference scores.

    Quality-aware threshold: low-quality parents require higher NLI scores to be retained.
    effective_threshold = threshold + (1 - parent_quality) * quality_penalty

    Special case: if parent is a text node and child is a section_header, force re-parenting
    even if the original score is not the lowest (text nodes should not parent section_headers).
    """
    from core.doc.feature_extract import classify_numbering_type

    fix_count = 0
    # Compute quality-aware threshold for each pair
    effective_thresholds = np.array(
        [
            threshold + (1.0 - _parent_quality(p)) * quality_penalty
            for p, _ in pairs_to_check
        ],
        dtype=np.float32,
    )

    # Identify cases requiring forced repair: text node parenting a section_header
    force_fix_indices = []
    for idx, (parent_node, child_node) in enumerate(pairs_to_check):
        if parent_node.label == "text" and child_node.label == "section_header":
            force_fix_indices.append(idx)

    # Merge low-score indices and forced-fix indices
    low_score_indices = np.where(scores < effective_thresholds)[0]
    all_fix_indices = set(low_score_indices.tolist()) | set(force_fix_indices)

    if len(all_fix_indices) == 0:
        return 0

    # 1. Collect candidate re-parent pairs for batch scoring
    reparent_tasks = []
    all_reparent_pairs = []

    for idx in sorted(all_fix_indices):
        parent_node, child_node = pairs_to_check[idx]
        ancestors = _collect_ancestor_chain(parent_node)
        if not ancestors:
            continue

        start_idx = len(all_reparent_pairs)
        for anc in ancestors:
            all_reparent_pairs.append((anc.text, child_node.text))

        reparent_tasks.append(
            {
                "orig_idx": idx,
                "child_node": child_node,
                "ancestors": ancestors,
                "range": (start_idx, len(ancestors)),
                "force_fix": idx in force_fix_indices,  # whether this is a forced fix
            }
        )

    if not all_reparent_pairs:
        return 0

    # 2. Second round of inference: find the best parent node
    reparent_scores = score_header_content_pairs(all_reparent_pairs)

    # 3. Apply modifications
    for task in reparent_tasks:
        child_node = task["child_node"]
        ancestors = task["ancestors"]
        start, count = task["range"]
        cand_scores = reparent_scores[start : start + count]
        orig_score = float(scores[task["orig_idx"]])
        is_force_fix = task.get("force_fix", False)

        # SEC Item Ceiling logic
        ceiling_idx = next(
            (
                j
                for j, anc in enumerate(ancestors)
                if classify_numbering_type(anc.text) == "sec_item"
            ),
            None,
        )

        # Improvement threshold: NLI total range ~9 points (-3~+6), 0.3 is ~3%, filters noise while allowing real fixes
        MIN_IMPROVEMENT = 0.3

        if ceiling_idx is not None:
            restricted_scores = cand_scores[: ceiling_idx + 1]
            best_idx = int(np.argmax(restricted_scores))
            # For forced fixes, accept as long as new parent is better than current text parent
            if (
                not is_force_fix
                and float(restricted_scores[best_idx]) - orig_score < MIN_IMPROVEMENT
            ):
                best_idx = ceiling_idx
        else:
            best_idx = int(np.argmax(cand_scores))
            # For forced fixes, accept as long as new parent is better than current text parent (lower threshold)
            if (
                not is_force_fix
                and float(cand_scores[best_idx]) - orig_score < MIN_IMPROVEMENT
            ):
                continue

        new_parent = ancestors[best_idx]
        if child_node.parent:
            child_node.parent.children = [
                c for c in child_node.parent.children if c is not child_node
            ]

        child_node.parent = new_parent
        new_parent.children.append(child_node)
        _update_depth(child_node, new_parent.depth + 1)
        fix_count += 1

    return fix_count


def verify_and_fix_parents(
    root: "Node",
    body_style: Dict[str, Any],
    threshold: float = 0.0,
) -> int:
    """Backward-compatible single-document verification function."""
    pairs = collect_parent_verification_tasks(root, body_style)
    if not pairs:
        return 0

    text_pairs = [(p.text, c.text) for p, c in pairs]
    scores = score_header_content_pairs(text_pairs)

    fix_count = apply_parent_fixes(pairs, scores, threshold)
    print(f"  [CE] Verified {len(pairs)} pairs, fixed {fix_count}")
    return fix_count


def _update_depth(node: "Node", new_depth: int) -> None:
    """Recursively update the depth of a node and all its descendants."""
    node.depth = new_depth
    for child in node.children:
        _update_depth(child, new_depth + 1)
