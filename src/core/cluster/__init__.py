"""core.cluster -- Document clustering module (Problem 2).

Canonical artifact workflow:
  S_sem (OT) + S_tfidf + S_tree
  -> weighted fusion (0.5/0.3/0.2)
  -> recursive spectral bisection + silhouette pruning
  -> single corpus-level LLM merge
"""

__all__ = [
    "common",
    "bisection",
]
