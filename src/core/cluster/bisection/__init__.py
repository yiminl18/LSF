"""core.cluster.bisection -- Recursive spectral bisection subpackage."""

from __future__ import annotations

from core.cluster.bisection.prepare_inputs import (
    main,
    prepare_canonical_inputs,
)
from core.cluster.bisection.data import load_all_data
from core.cluster.bisection.similarity import (
    build_heading_tfidf_similarity,
    build_tree_shape_similarity,
)
from core.cluster.bisection.llm_merge import (
    llm_merge_clusters,
    sample_cluster_headings,
)
from core.cluster.bisection.metrics import (
    class_precision,
    class_recall,
    hungarian_map,
    hungarian_map_from_leaves,
    merge_periodic,
    save_assignments,
    save_confusion_csv,
)
from core.cluster.bisection.pipeline import (
    main as pipeline_main,
    run_recursive_fused,
)
from core.cluster.bisection.recursive import recursive_bisect_pruned

__all__ = [
    # data
    "load_all_data",
    "prepare_canonical_inputs",
    # recursive
    "recursive_bisect_pruned",
    # metrics
    "hungarian_map",
    "hungarian_map_from_leaves",
    "class_recall",
    "class_precision",
    "merge_periodic",
    "save_confusion_csv",
    "save_assignments",
    # llm_merge
    "sample_cluster_headings",
    "llm_merge_clusters",
    # similarity helpers
    "build_heading_tfidf_similarity",
    "build_tree_shape_similarity",
    # pipeline
    "run_recursive_fused",
    "pipeline_main",
    "main",
]
