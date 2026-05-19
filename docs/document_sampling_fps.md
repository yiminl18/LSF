# Document Sampling via Farthest-Point Traversal

This document specifies a hyperparameter-free sampling strategy for selecting a diverse subset of documents from a corpus, such that every underlying cluster is represented by at least one sampled instance. The approach uses farthest-point sampling (also known as farthest-first traversal or Gonzalez's algorithm) on a per-(query, doc) similarity-curve feature space.

The motivation is to replace earlier threshold-based dedup samplers (which required tuning a similarity cutoff `τ`) with a deterministic algorithm whose only stopping signal is an elbow read off the data itself.

---

## 1. Goal

Pick a subset `R ⊆ D` of the corpus that:

- Contains at least one representative from every underlying cluster (where "cluster" means "documents whose answer to a given query lives in similar locations").
- Uses no tunable hyperparameter — no threshold, no patience window, no `ε` or `δ`.
- Provides a deterministic guarantee on cluster coverage when the data is geometrically well-separated.

The hidden cluster structure is encoded geometrically: documents whose `v_d` vectors (defined below) are close share an answer-location pattern.

---

## 2. Feature representation per (query, doc)

For a fixed query `q`, build one vector per document:

1. Embed `q` once with a sentence/passage embedding model (e.g. `text-embedding-3-large` or a local BGE variant).
2. For each document `d ∈ D`:
   - Chunk `d` into `N_d` chunks. Either fixed-size (~256 tokens) or structural (use `doc["texts"]` spans).
   - Embed each chunk.
   - Compute the similarity sequence `s_d(i) = cos(emb(q), emb(chunk_i))` for `i = 1, …, N_d`.
3. **Position-normalise.** Rescale chunk index to `[0, 1]` and resample to a fixed length `L` (e.g. `L = 50`) by averaging within bins.
4. **Contrast-normalise.** Subtract the per-document median of `s_d` and divide by the IQR. This removes the absolute similarity level (which varies with how related `q` is to `d` overall) and preserves the curve **shape**.

The result is a fixed-dimensional vector `v_d ∈ ℝ^L` per document. Documents whose answers live in similar locations have `v_d`'s with peaks at similar positions.

---

## 3. Distance function

Use cosine distance on the contrast-normalised vectors:

```
dist(d_a, d_b) = 1 - cos(v_a, v_b)
                = 1 - (v_a · v_b) / (||v_a|| · ||v_b||)
```

Cosine on shape-normalised vectors emphasises *which positions in the document the query matches*, not how strongly it matches. Two docs where the query's answer-relevant content sits at the same position-bin produce parallel `v_d`'s and small cosine distance, regardless of absolute similarity magnitude.

Alternative: 1-Wasserstein distance on the (renormalised-to-distribution) curves. More principled distributionally; rarely needed in practice. Cosine is the default.

---

## 4. The sampling algorithm

Farthest-point sampling (Gonzalez 1985 / farthest-first traversal):

```
INPUT  : corpus D, distance dist(·, ·)
OUTPUT : ordered sample R = [d_1, d_2, …, d_K], gap sequence [g_1, g_2, …]

1. d_1 ← any document   (random, or argmax_d dist(d, centroid of D))
2. R ← [d_1]
3. gaps ← []
4. Loop:
       d_next ← argmax_{d ∉ R}  min_{r ∈ R}  dist(d, r)
       g_next ← min_{r ∈ R}  dist(d_next, r)
       R.append(d_next)
       gaps.append(g_next)
   until stopping rule (see §5).
5. Return R, gaps.
```

Each new pick is, by construction, the document farthest in cosine distance from everything sampled so far. No threshold; no patience window; the algorithm just walks the most-diverse-first traversal of the corpus.

**Complexity.** Each iteration is `O(|D|)` distance evaluations (one per remaining doc against the current `R`). Total cost `O(K · |D|)` for `K` picks, which is trivial for typical `|D| < 10^4`.

---

## 5. Stopping rule (elbow on the gap sequence)

The gap sequence `g_1 ≥ g_2 ≥ … ≥ g_i ≥ …` is monotonically non-increasing by construction.

- While the algorithm is still discovering *new clusters*, each new pick must come from an as-yet-unrepresented cluster, so `g_i` is bounded below by the inter-cluster separation `δ_inter`.
- Once every cluster has at least one representative in `R`, every subsequent pick lies *within* an already-represented cluster, so `g_i ≤ δ_intra` (the within-cluster diameter).

When clusters are well-separated (`δ_inter > 2·δ_intra`), the transition from "discovery" to "refinement" produces a sharp drop in `g_i`. Detect it:

**Default rule.** Stop after step `i` if `g_{i+1} < 0.5 · g_i`. The first such `i` is the estimated number of clusters `k̂`.

**Alternative.** Plot `g_i` vs `i`; the elbow is visually obvious when separation holds. For automated pipelines the 0.5× ratio rule works well; for diagnostic / exploratory runs, plot.

No threshold has to be picked ahead of time — the elbow is a property of the data.

---

## 6. Guarantees

### 6.1 The cluster-coverage theorem

Suppose the corpus has `k` true clusters such that:

- **Intra-cluster diameter** ≤ `δ_intra` (the farthest pair within any one cluster).
- **Inter-cluster separation** ≥ `δ_inter` (the closest pair from different clusters).

If `δ_inter > 2 · δ_intra` (well-separated), then farthest-point sampling run for `k` steps deterministically picks at least one representative from every cluster.

**Proof sketch.** Gonzalez's theorem gives, for `|R| = k`:

```
max_d min_{r ∈ R} dist(d, r) ≤ 2 · OPT_k ≤ 2 · δ_intra
```

(`OPT_k ≤ δ_intra` because the optimal `k`-cover can pick one point per cluster, and every doc is within `δ_intra` of its cluster's centre.) So every doc is within `2·δ_intra` of some `r ∈ R`. But docs from different clusters are separated by more than `2·δ_intra`. So no single `r` can cover two clusters, and `R` must contain at least one point per cluster. ∎

### 6.2 The elbow exists and is detectable

Under the same well-separation:

- For `i ≤ k`: `g_i` can be as large as `δ_inter` (jumping between clusters).
- For `i = k+1`: `g_{k+1} ≤ δ_intra` (any new pick is within an existing cluster).

Hence `g_k / g_{k+1} ≥ δ_inter / δ_intra > 2`. The default rule (`g_{i+1} < 0.5 · g_i`) fires exactly at `i = k`, so it identifies the correct cluster count without external input.

### 6.3 2-approximation to k-center

For any chosen `K`, the picked `R` satisfies

```
max_d min_{r ∈ R} dist(d, r) ≤ 2 · OPT_K
```

where `OPT_K` is the optimal k-center cost. This is the standard Gonzalez guarantee, used as the underlying lemma for §6.1.

### 6.4 Outlier prioritisation

A singleton or rare-cluster document maximises the max-min distance the moment it's the only unrepresented point; FPS picks it as soon as that condition holds. In particular, rare clusters are typically picked *early*, not late — the opposite of random sampling's bias.

---

## 7. Failure mode: clusters that touch

If `δ_inter ≤ 2 · δ_intra` (clusters touch or overlap), no deterministic distance-based algorithm can recover one rep per cluster. FPS in that regime can:

- Cover two near-touching clusters with a single representative (under-coverage).
- Split a single elongated cluster into two representatives (over-coverage).

This is an **identifiability limit** of the chosen feature space and distance — not a flaw of FPS. The remedy is to enrich the features (use a stronger embedding model, add document-property features, condition on more queries) so the clusters are pulled apart in vector space. Verify separation once on a small labelled subset before deploying.

---

## 8. Comparison to alternatives

| Aspect | Random | Threshold dedup (τ) | Farthest-point sampling |
|---|---|---|---|
| Hyperparameter | sample size | similarity threshold `τ` | none (elbow self-detects) |
| Cluster recovery | probabilistic | probabilistic (patience-bounded) | deterministic under well-separation |
| Rare cluster handling | misses with prob `(1−s/n)^k` | picks if encountered | picks first |
| Distance information used | none (binary include/exclude) | binary (`same` predicate) | full continuous distance |
| Per-pick complexity | `O(1)` | `O(|R|)` | `O(|D|)` |
| Determinism | no | depends on shuffle | yes (given starting point) |

For continuous-vector feature spaces like the `v_d` cosine setup, FPS is strictly preferable: it extracts more information from the data, has no knob, and gives a stronger guarantee.

---

## 9. Practical recipe

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def farthest_point_sampling(V: np.ndarray,
                            stop_ratio: float = 0.5,
                            max_K: int | None = None,
                            seed: int = 0):
    """
    V          : (N, L) contrast-normalised similarity-curve vectors
    stop_ratio : stop at first i with g_{i+1} < stop_ratio · g_i  (default 0.5)
    max_K      : optional hard cap on |R|

    Returns
    -------
    indices    : list[int], the order in which docs were picked
    gaps       : list[float], gap g_i at each pick (g_1 is None for the seed)
    """
    N = V.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(N))
    indices = [first]
    gaps = [None]

    # min distance from every doc to the current sample
    D = cosine_distances(V, V[first:first+1]).reshape(-1)

    while True:
        d_next = int(np.argmax(D))
        g = float(D[d_next])

        if len(indices) >= 2 and g < stop_ratio * gaps[-1]:
            break
        if max_K is not None and len(indices) >= max_K:
            break

        indices.append(d_next)
        gaps.append(g)

        # update D to be min distance to the *new* sample
        D = np.minimum(D, cosine_distances(V, V[d_next:d_next+1]).reshape(-1))

    return indices, gaps
```

Usage:

```python
V = build_similarity_curve_vectors(corpus, query, embed_fn, L=50)
indices, gaps = farthest_point_sampling(V, stop_ratio=0.5)
sample = [corpus[i] for i in indices]
```

---

## 10. Validation

Before deploying, calibrate that the chosen feature space and distance satisfy `δ_inter > 2 · δ_intra`:

1. Take a labelled subset of 20–50 documents where you know the true cluster (e.g. by bucketing the ground-truth answer location by `(page_band, section_path)`).
2. Compute the pairwise cosine-distance matrix between all `v_d` in that subset.
3. Compute `δ_intra = max_{(a,b) in same cluster} dist(v_a, v_b)`.
4. Compute `δ_inter = min_{(a,b) in different clusters} dist(v_a, v_b)`.
5. If `δ_inter > 2 · δ_intra`, the deterministic cluster-coverage theorem applies; deploy FPS with confidence.
6. If the ratio is smaller than 2 but greater than 1, FPS will still typically pick one rep per cluster but the guarantee is empirical, not deterministic. Acceptable for most uses.
7. If `δ_inter ≤ δ_intra`, the features are not separating clusters; enrich them (better embeddings, more queries, document-property features) before sampling.

A second sanity check after sampling: compute the gap-ratio `g_k / g_{k+1}` and confirm it exceeds 2. If it does, the elbow detection is robust on this run. If not, hand-inspect the curve.

---

## 11. Output artefact

Persist the sample and the gap trace:

```json
{
  "query": "...",
  "feature_dim": 50,
  "distance": "cosine_on_contrast_normalised",
  "K_picked": 7,
  "stop_ratio": 0.5,
  "indices": [...],
  "doc_names": ["AMCOR_2019_10K", "BOEING_2018_10K", ...],
  "gaps": [null, 0.62, 0.58, 0.51, 0.48, 0.45, 0.42, 0.19],
  "elbow_at": 7,
  "gap_ratio_at_elbow": 2.21,
  "separation_check": {
    "delta_intra_observed": 0.18,
    "delta_inter_observed": 0.46,
    "ratio": 2.56,
    "deterministic_guarantee_holds": true
  }
}
```

The `gaps` array is the diagnostic that lets you re-inspect the elbow later. `separation_check` records the validation from §10.

---

## 12. Summary

Farthest-point sampling on cosine distance over contrast-normalised similarity-curve vectors gives a hyperparameter-free way to sample a corpus such that every cluster is represented. The deterministic cluster-coverage guarantee holds whenever the feature space well-separates the clusters (`δ_inter > 2 · δ_intra`), a property you verify once per dataset. The natural elbow in the max-min gap sequence both identifies the number of clusters and tells the algorithm when to stop, so the only "knob" is the universally-default ratio `g_{i+1}/g_i < 0.5` for elbow detection — and even that is just a tighter spelling of the well-separation assumption already required for the guarantee.

For the LSF use case — picking a representative subset of financial filings whose query-answer locations span the corpus's template diversity — this is the recommended sampler.
