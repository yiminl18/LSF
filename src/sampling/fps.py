"""Farthest-Point Sampling (Gonzalez 1985 / farthest-first traversal).

Implements §4 + §5 + §9 of docs/document_sampling_fps.md:
  - Iterative max-min distance picks
  - Elbow stopping rule on the gap sequence

Distance is cosine on contrast-normalised similarity-curve vectors;
caller passes those vectors in directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def farthest_point_sampling(
    V:           np.ndarray,
    stop_ratio:  float = 0.5,
    max_K:       int | None = None,
    seed:        int = 0,
    first_index: int | None = None,
) -> tuple[list[int], list[float | None]]:
    """Run FPS on the row vectors of V.

    Parameters
    ----------
    V          : (N, D) array of feature vectors (one per doc).
    stop_ratio : stop at the first i where g_{i+1} < stop_ratio · g_i  (i >= 2).
                 Set to 0.0 to disable elbow stopping (only max_K can stop the loop).
    max_K      : optional hard cap on |R|. None = no cap.
    seed       : RNG seed for picking the first doc (when first_index is None).
    first_index: explicit index for the first pick. If None, picks uniformly at
                 random from [0, N).

    Returns
    -------
    indices : ordered list of picked doc indices.
    gaps    : per-pick gap g_i (the max-min distance at the moment of picking).
              gaps[0] is None (the seed has no gap defined).
    """
    N = int(V.shape[0])
    if N == 0:
        return [], []

    rng = np.random.default_rng(seed)
    first = int(first_index) if first_index is not None else int(rng.integers(N))

    indices: list[int] = [first]
    gaps:    list[float | None] = [None]

    # D[d] = min distance from doc d to the current sample R
    D = cosine_distances(V, V[first:first + 1]).reshape(-1)

    while True:
        d_next = int(np.argmax(D))
        g = float(D[d_next])

        # Elbow stop: fires only after we have at least 2 gaps to compare
        if len(indices) >= 2 and stop_ratio > 0 and gaps[-1] is not None:
            if g < stop_ratio * gaps[-1]:
                break

        # Hard cap
        if max_K is not None and len(indices) >= max_K:
            break

        # Edge: if everyone is at distance 0, we can't pick anything useful
        if g <= 0.0:
            break

        indices.append(d_next)
        gaps.append(g)

        # Update min-distance with the new sample point
        D = np.minimum(
            D, cosine_distances(V, V[d_next:d_next + 1]).reshape(-1)
        )

    return indices, gaps
