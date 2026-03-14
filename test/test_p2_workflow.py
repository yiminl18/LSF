"""
test_p2_workflow.py — Educational walkthrough of the LSF Problem 2 pipeline.

LSF Problem 2 solves *unsupervised document type clustering* for a corpus
of financial PDFs.  Given ~600 documents with no labels, the system
discovers document types (10-K, 10-Q, 8-K, EARNINGS, CHI) by fusing
three complementary similarity signals and applying recursive spectral
bisection with silhouette-based pruning.

The retained P2 pipeline:
  1. Compute semantic similarity (S_sem) via Optimal Transport on embeddings
  2. Compute heading text similarity (S_tfidf) via TF-IDF cosine
  3. Compute tree shape similarity (S_tree) via RBF kernel on structural features
  4. Fuse: S = w_sem * S_sem + w_tfidf * S_tfidf + w_tree * S_tree
  5. Recursive spectral bisection with silhouette pruning
  6. Run one corpus-level LLM merge over the pruned clusters
  7. Evaluate: NMI, ARI, per-class recall via Hungarian mapping

This file tests the *algorithmic concepts* using synthetic data and mocks,
so it runs without datasets, API keys, or the full clustering pipeline.

Run:
    pytest test/test_p2_workflow.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestRetainedPackageSurface(unittest.TestCase):
    """Guard the artifact-facing P2 package surface."""

    def test_cluster_package_does_not_export_analysis(self):
        """The artifact keeps only the canonical common + bisection surface."""
        import core.cluster

        self.assertEqual(core.cluster.__all__, ["common", "bisection"])


# ---------------------------------------------------------------------------
# 1. Similarity matrix construction
# ---------------------------------------------------------------------------

class TestSimilarityConstruction(unittest.TestCase):
    """Test the three similarity matrices that form the P2 fusion input.

    The core insight of P2 is that no single similarity measure captures
    all aspects of document type.  By fusing three complementary signals,
    the system achieves NMI > 0.85 where any single signal achieves < 0.70.

    The three similarity matrices:
      S_sem   — Semantic similarity via Optimal Transport on embeddings
      S_tfidf — Heading text similarity via TF-IDF cosine similarity
      S_tree  — Tree shape similarity via RBF kernel on structural features
    """

    def test_semantic_similarity_via_cosine(self):
        """S_sem captures document-level semantic similarity.

        Each document is represented by its header embeddings (from OpenAI
        or OpenRouter).  The pairwise similarity uses Optimal Transport (OT)
        to align header sets across documents, accounting for the fact that
        documents may have different numbers of headers.

        For this test, we simulate a simpler cosine-based approach to
        demonstrate the NxN similarity matrix construction pattern.
        """
        n_docs = 4
        embed_dim = 3

        # Simulate document embeddings (mean-pooled from headers)
        doc_embeddings = np.array([
            [1.0, 0.0, 0.0],   # doc 0: type A
            [0.9, 0.1, 0.0],   # doc 1: type A (similar to doc 0)
            [0.0, 1.0, 0.0],   # doc 2: type B
            [0.0, 0.9, 0.1],   # doc 3: type B (similar to doc 2)
        ])

        # Build NxN cosine similarity matrix
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        normalized = doc_embeddings / norms
        S_sem = normalized @ normalized.T

        # Verify properties
        self.assertEqual(S_sem.shape, (n_docs, n_docs))
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(S_sem), 1.0, atol=1e-6)
        # Same-type docs should be more similar than cross-type
        self.assertGreater(S_sem[0, 1], S_sem[0, 2])
        self.assertGreater(S_sem[2, 3], S_sem[0, 3])

    def test_tfidf_heading_similarity(self):
        """S_tfidf captures domain-agnostic heading vocabulary patterns.

        Each document's headings are concatenated into a single "heading
        document".  TF-IDF automatically discovers discriminative terms:
        e.g., "Management Discussion" appears in 10-K but not 8-K, so it
        gets high TF-IDF weight.

        The TfidfVectorizer uses max_df=0.9 (ignore terms in >90% of docs)
        and min_df=2 (ignore terms appearing in only 1 doc).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Simulate heading documents for 4 docs
        heading_docs = [
            "management discussion analysis financial statements notes",  # 10-K style
            "management discussion analysis risk factors",                # 10-K style
            "current report material event exhibits",                     # 8-K style
            "current report item exhibits signature",                     # 8-K style
        ]

        vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(heading_docs)
        S_tfidf = cosine_similarity(tfidf_matrix)
        np.clip(S_tfidf, 0.0, 1.0, out=S_tfidf)

        # Same-type docs should cluster together
        self.assertGreater(S_tfidf[0, 1], S_tfidf[0, 2],
                           "10-K docs should be more similar to each other")
        self.assertGreater(S_tfidf[2, 3], S_tfidf[0, 3],
                           "8-K docs should be more similar to each other")

    def test_tree_shape_similarity_via_rbf(self):
        """S_tree captures structural fingerprint similarity via RBF kernel.

        Per-document structural features:
          - Level-wise node count distribution (normalized histogram)
          - Leaf/internal node ratio
          - Average branching factor
          - Depth distribution entropy

        RBF kernel is chosen over cosine similarity because:
          1. Features are mixed-scale (counts vs ratios vs entropy)
          2. RBF captures non-linear similarity patterns
          3. gamma=1/n_features is a robust default

        10-K documents tend to have deep trees (depth 4-6) with many
        sections, while 8-K documents are shallow (depth 1-2).
        """
        from sklearn.metrics.pairwise import rbf_kernel

        # Simulate structural features for 4 docs
        features = np.array([
            [0.1, 0.3, 0.4, 0.2, 0.3, 3.5, 1.8],  # deep tree (10-K)
            [0.1, 0.2, 0.5, 0.2, 0.35, 3.2, 1.9],  # deep tree (10-K)
            [0.6, 0.4, 0.0, 0.0, 0.7, 1.5, 0.8],   # shallow tree (8-K)
            [0.5, 0.5, 0.0, 0.0, 0.65, 1.8, 0.7],   # shallow tree (8-K)
        ])

        S_tree = rbf_kernel(features, gamma=1.0 / features.shape[1])

        # Same-type docs should be more similar
        self.assertGreater(S_tree[0, 1], S_tree[0, 2])
        self.assertGreater(S_tree[2, 3], S_tree[0, 3])


# ---------------------------------------------------------------------------
# 2. Weighted fusion
# ---------------------------------------------------------------------------

class TestWeightedFusion(unittest.TestCase):
    """Test the weighted fusion of three similarity matrices.

    The fused similarity matrix is a convex combination:
        S = w_sem * S_sem + w_tfidf * S_tfidf + w_tree * S_tree

    Default weights: w_sem=0.5, w_tfidf=0.3, w_tree=0.2

    Why these weights?
      - Semantic similarity (0.5) is the strongest single signal because
        OT-based embedding comparison captures deep content patterns.
      - TF-IDF heading similarity (0.3) adds discriminative vocabulary
        that embeddings may smooth over (e.g., exact section titles).
      - Tree shape (0.2) provides structural signal independent of content,
        useful when documents share vocabulary but differ structurally.

    The artifact fixes the weights to 0.5 / 0.3 / 0.2 so the workflow stays
    canonical and reproducible.
    """

    def test_fusion_weights_sum_to_one(self):
        """Weights must form a convex combination (sum to 1.0)."""
        w_sem, w_tfidf, w_tree = 0.5, 0.3, 0.2
        self.assertAlmostEqual(w_sem + w_tfidf + w_tree, 1.0)

    def test_fusion_computation(self):
        """Fused matrix should be a weighted average of input matrices.

        The fusion preserves the range [0, 1] because each input matrix
        has values in [0, 1] and the weights form a convex combination.
        """
        n = 3
        S_sem = np.array([[1.0, 0.8, 0.2],
                          [0.8, 1.0, 0.3],
                          [0.2, 0.3, 1.0]])
        S_tfidf = np.array([[1.0, 0.9, 0.1],
                            [0.9, 1.0, 0.2],
                            [0.1, 0.2, 1.0]])
        S_tree = np.array([[1.0, 0.7, 0.3],
                           [0.7, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])

        w_sem, w_tfidf, w_tree = 0.5, 0.3, 0.2
        S_fused = w_sem * S_sem + w_tfidf * S_tfidf + w_tree * S_tree

        # Manual check for S_fused[0,1]
        expected_01 = 0.5 * 0.8 + 0.3 * 0.9 + 0.2 * 0.7  # 0.4 + 0.27 + 0.14 = 0.81
        self.assertAlmostEqual(S_fused[0, 1], expected_01)

        # Diagonal should remain 1.0
        np.testing.assert_allclose(np.diag(S_fused), 1.0, atol=1e-10)

        # All values should be in [0, 1]
        self.assertTrue(np.all(S_fused >= 0.0))
        self.assertTrue(np.all(S_fused <= 1.0))

    def test_semantic_dominance(self):
        """Semantic signal should dominate when content clearly separates types.

        If S_sem clearly separates documents but S_tfidf and S_tree are
        noisy, the fused matrix should still produce clean clusters because
        w_sem=0.5 dominates.  This robustness is why the system works even
        when structural parsing has errors.
        """
        n = 4
        # S_sem clearly separates: docs 0,1 vs docs 2,3
        S_sem = np.array([[1.0, 0.9, 0.1, 0.1],
                          [0.9, 1.0, 0.1, 0.1],
                          [0.1, 0.1, 1.0, 0.9],
                          [0.1, 0.1, 0.9, 1.0]])
        # S_tfidf is noisy
        S_noise = 0.5 * np.ones((n, n))
        np.fill_diagonal(S_noise, 1.0)

        S_fused = 0.5 * S_sem + 0.3 * S_noise + 0.2 * S_noise

        # Same-type similarity should still be higher
        self.assertGreater(S_fused[0, 1], S_fused[0, 2])


# ---------------------------------------------------------------------------
# 3. Recursive bisection with silhouette pruning
# ---------------------------------------------------------------------------

class TestRecursiveBisection(unittest.TestCase):
    """Test recursive spectral bisection with silhouette-based pruning.

    The SOTA clustering method recursively splits the corpus using
    SpectralClustering(k=2) at each level.  At each split, the silhouette
    score is computed on the distance matrix (D = 1 - S).  If silhouette
    < threshold (default 0.06), the split is rejected and the node becomes
    a leaf cluster.

    Why recursive bisection instead of flat k-means?
      1. Unknown k: we don't know how many document types exist a priori
      2. Hierarchical structure: some types are similar (10-K, 10-Q are both
         periodic filings) and should split later in the hierarchy
      3. Automatic stopping: silhouette pruning determines k adaptively

    Parameters:
      - min_cluster_size = 15 (don't split clusters smaller than 30 docs)
      - sil_threshold = 0.06 (conservative — prefers more splits)
      - max_depth = 6 (limits recursion for safety)
    """

    def test_silhouette_pruning_stops_bad_splits(self):
        """Low silhouette means the 2-way split is not meaningful.

        Silhouette score ranges from -1 to 1:
          > 0.5: strong cluster structure
          0.25-0.5: reasonable structure
          < 0.25: weak or artificial structure
          < 0: data point is likely in the wrong cluster

        The threshold 0.06 is very conservative — it only stops splits that
        are clearly meaningless.  This prefers over-splitting (which LLM
        merge can fix later) over under-splitting (which loses information).
        """
        sil_threshold = 0.06

        # A good split
        good_sil = 0.35
        self.assertTrue(good_sil >= sil_threshold, "Good split should proceed")

        # A bad split (essentially random)
        bad_sil = 0.02
        self.assertFalse(bad_sil >= sil_threshold, "Bad split should be pruned")

    def test_min_cluster_size_constraint(self):
        """Clusters smaller than 2 * min_cluster_size cannot be bisected.

        This prevents creating tiny clusters from statistical noise.
        With min_cluster_size=15, a cluster of 29 docs will not be split
        because each child would have < 15 docs.
        """
        min_cluster_size = 15
        cluster_size = 29

        can_split = cluster_size >= min_cluster_size * 2
        self.assertFalse(can_split)

        # A cluster of 30 can be split
        can_split_30 = 30 >= min_cluster_size * 2
        self.assertTrue(can_split_30)

    def test_distance_matrix_from_similarity(self):
        """Distance matrix D = clip(1 - S, 0) with zero diagonal.

        SpectralClustering uses a similarity matrix, but silhouette_score
        needs a distance (dissimilarity) matrix.  The conversion is:
            D = 1 - S, clipped to [0, inf), diagonal set to 0.

        Clipping is needed because S can slightly exceed 1.0 due to
        floating-point arithmetic in RBF/cosine computations.
        """
        S = np.array([[1.0, 0.8, 0.2],
                      [0.8, 1.0, 0.3],
                      [0.2, 0.3, 1.0]])

        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)

        # D[0,1] = 1 - 0.8 = 0.2
        self.assertAlmostEqual(D[0, 1], 0.2)
        # D[0,2] = 1 - 0.2 = 0.8
        self.assertAlmostEqual(D[0, 2], 0.8)
        # Diagonal is zero
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_recursive_bisect_produces_leaf_assignments(self):
        """The output is a dict mapping doc_id → leaf_label string.

        Each leaf in the recursion tree gets a label like 'L0', 'R1', 'L0L0'
        indicating its path through the binary tree.  These labels are later
        mapped to ground-truth types via Hungarian matching.
        """
        # Simulate a simple recursive bisection result
        assignments = {
            "doc_001": "L0",   # left branch, leaf 0
            "doc_002": "L0",
            "doc_003": "R1",   # right branch, leaf 1
            "doc_004": "R1",
            "doc_005": "L0R0", # left then right
        }

        # All docs should have an assignment
        self.assertEqual(len(assignments), 5)
        # Assignments should be strings
        for label in assignments.values():
            self.assertIsInstance(label, str)


# ---------------------------------------------------------------------------
# 4. LLM merge interface
# ---------------------------------------------------------------------------

class TestLLMMerge(unittest.TestCase):
    """Test the LLM-guided cluster merging concept.

    After recursive bisection, some leaf clusters may represent the same
    document type (over-splitting).  The LLM merge step:
      1. Samples representative headings from each leaf cluster
      2. Asks an LLM to identify which clusters should be merged
      3. Applies the merge to produce final cluster labels

    This is a post-processing refinement step.  The pipeline works without
    it (silhouette pruning alone achieves NMI > 0.80), but LLM merge can
    improve interpretability and fix over-splits that silhouette missed.

    The LLM prompt provides cluster headings and asks for merge decisions
    based on document type similarity, not content similarity.
    """

    def test_heading_sampling_deduplicates(self):
        """sample_cluster_headings should deduplicate heading text.

        Many documents share identical headings (e.g., "Table of Contents",
        "Signatures").  Deduplication ensures the LLM sees diverse headings
        rather than repetitive boilerplate.
        """
        raw_headings = [
            "Table of Contents",
            "table of contents",  # case-insensitive duplicate
            "Management Discussion",
            "Risk Factors",
            "Table of Contents",  # exact duplicate
        ]

        # Simulate deduplication logic from llm_merge.py
        seen = set()
        unique = []
        for h in raw_headings:
            h_lower = h.strip().lower()
            if h_lower not in seen:
                seen.add(h_lower)
                unique.append(h.strip())

        self.assertEqual(len(unique), 3)
        self.assertIn("Table of Contents", unique)
        self.assertIn("Management Discussion", unique)
        self.assertIn("Risk Factors", unique)

    def test_merge_reduces_cluster_count(self):
        """Merging over-split clusters should reduce the number of unique labels.

        Example: recursive bisection produces 7 leaf clusters, but ground
        truth has 5 types.  LLM identifies that clusters 'L0L0' and 'L0L1'
        are both '10-K annual reports' and merges them.
        """
        # Before merge: 7 clusters
        pre_merge = {
            "doc_001": "L0L0",
            "doc_002": "L0L1",
            "doc_003": "R0",
            "doc_004": "R1L0",
            "doc_005": "R1L1",
            "doc_006": "R1R0",
            "doc_007": "R1R1",
        }
        unique_before = len(set(pre_merge.values()))
        self.assertEqual(unique_before, 7)

        # LLM decides: L0L0 + L0L1 → "annual", R1L0 + R1L1 → "quarterly"
        merge_map = {
            "L0L0": "annual",
            "L0L1": "annual",
            "R0": "current",
            "R1L0": "quarterly",
            "R1L1": "quarterly",
            "R1R0": "earnings",
            "R1R1": "other",
        }
        post_merge = {doc: merge_map[label] for doc, label in pre_merge.items()}
        unique_after = len(set(post_merge.values()))

        self.assertEqual(unique_after, 5)
        self.assertLess(unique_after, unique_before)


# ---------------------------------------------------------------------------
# 5. Metrics: NMI, ARI, per-class recall via Hungarian mapping
# ---------------------------------------------------------------------------

class TestClusteringMetrics(unittest.TestCase):
    """Test clustering evaluation metrics.

    Since clustering is unsupervised, cluster labels are arbitrary integers.
    To compare against ground truth, we need:
      1. Hungarian mapping: optimally assign cluster IDs to GT types
      2. NMI: Normalized Mutual Information (0=random, 1=perfect)
      3. ARI: Adjusted Rand Index (0=random, 1=perfect, can be negative)
      4. Per-class recall: fraction of each GT type correctly clustered

    NMI is the primary metric because it's invariant to the number of
    clusters (unlike accuracy, which favors the trivial k=1 solution).
    """

    def test_hungarian_mapping_concept(self):
        """Hungarian algorithm finds optimal cluster-to-type assignment.

        Given a contingency matrix C[cluster_i, type_j] = count of docs in
        cluster i with GT type j, the Hungarian algorithm minimizes the
        total assignment cost (equivalently, maximizes overlap).

        This is implemented in metrics.py using scipy.optimize.linear_sum_assignment.
        """
        from scipy.optimize import linear_sum_assignment

        # Contingency matrix: 3 clusters, 3 types
        # Cluster 0 has 8 docs of type B, 2 of type A
        # Cluster 1 has 9 docs of type A, 1 of type C
        # Cluster 2 has 7 docs of type C, 3 of type B
        C = np.array([
            [2, 8, 0],  # cluster 0
            [9, 0, 1],  # cluster 1
            [0, 3, 7],  # cluster 2
        ])

        # Hungarian minimizes cost → use negative contingency
        row_ind, col_ind = linear_sum_assignment(-C.astype(float))

        # Optimal mapping: cluster 0→type B, cluster 1→type A, cluster 2→type C
        type_order = ["A", "B", "C"]
        mapping = {r: type_order[c] for r, c in zip(row_ind, col_ind)}

        self.assertEqual(mapping[0], "B")
        self.assertEqual(mapping[1], "A")
        self.assertEqual(mapping[2], "C")

    def test_nmi_perfect_clustering(self):
        """NMI = 1.0 when clustering perfectly matches ground truth."""
        from sklearn.metrics import normalized_mutual_info_score

        gt = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # Perfect clustering (labels may differ but partition matches)
        pred = [2, 2, 2, 0, 0, 0, 1, 1, 1]

        nmi = normalized_mutual_info_score(gt, pred)
        self.assertAlmostEqual(nmi, 1.0, places=5)

    def test_nmi_random_clustering(self):
        """NMI ≈ 0.0 when clustering is random (no mutual information)."""
        from sklearn.metrics import normalized_mutual_info_score

        gt = [0] * 50 + [1] * 50
        np.random.seed(42)
        pred = np.random.randint(0, 2, 100).tolist()

        nmi = normalized_mutual_info_score(gt, pred)
        # Random clustering should have very low NMI
        self.assertLess(nmi, 0.1)

    def test_ari_properties(self):
        """ARI = 1.0 for perfect, ~0 for random, can be negative.

        ARI adjusts for chance: two random clusterings have expected ARI = 0.
        This makes ARI more informative than raw accuracy for imbalanced
        cluster sizes.
        """
        from sklearn.metrics import adjusted_rand_score

        gt = [0, 0, 0, 1, 1, 1]
        perfect = [1, 1, 1, 0, 0, 0]  # same partition, different labels
        ari_perfect = adjusted_rand_score(gt, perfect)
        self.assertAlmostEqual(ari_perfect, 1.0)

        # Worst case: every point in its own cluster
        singleton = [0, 1, 2, 3, 4, 5]
        ari_bad = adjusted_rand_score(gt, singleton)
        self.assertLess(ari_bad, 0.5)

    def test_class_recall_computation(self):
        """Per-class recall measures how well each GT type is recovered.

        recall(cls) = TP / (TP + FN)

        For document clustering, this answers: "What fraction of 10-Q
        documents did we correctly identify as 10-Q?"

        This is critical for the SEC use case: failing to identify a 10-Q
        means missing quarterly financial data.
        """
        gt   = ["10K", "10K", "10K", "10Q", "10Q", "8K", "8K", "8K"]
        pred = ["10K", "10K", "10Q", "10Q", "10Q", "8K", "8K", "10K"]

        def class_recall(gt_list, pred_list, cls):
            tp = sum(1 for g, p in zip(gt_list, pred_list) if g == cls and p == cls)
            fn = sum(1 for g, p in zip(gt_list, pred_list) if g == cls and p != cls)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # 10K: 2 correct out of 3 → recall = 2/3
        self.assertAlmostEqual(class_recall(gt, pred, "10K"), 2.0 / 3.0)
        # 10Q: 2 correct out of 2 → recall = 1.0
        self.assertAlmostEqual(class_recall(gt, pred, "10Q"), 1.0)
        # 8K: 2 correct out of 3 → recall = 2/3
        self.assertAlmostEqual(class_recall(gt, pred, "8K"), 2.0 / 3.0)

    def test_merge_periodic_helper(self):
        """merge_periodic maps 10K/10Q → PERIODIC for coarser evaluation.

        Some analyses group periodic filings together (10-K annual + 10-Q
        quarterly) because they share structural patterns.  This 3-class
        view (PERIODIC, 8K, CHI) shows higher NMI because the 10K/10Q
        distinction is the hardest to learn.
        """
        labels = ["10K", "10Q", "8K", "10K", "EARNINGS", "CHI"]
        merged = ["PERIODIC" if t in ("10K", "10Q") else t for t in labels]

        self.assertEqual(merged, ["PERIODIC", "PERIODIC", "8K",
                                   "PERIODIC", "EARNINGS", "CHI"])
        # Unique count drops from 5 to 4
        self.assertEqual(len(set(labels)), 5)
        self.assertEqual(len(set(merged)), 4)


# ---------------------------------------------------------------------------
# 6. End-to-end concept: full pipeline flow
# ---------------------------------------------------------------------------

class TestPipelineFlow(unittest.TestCase):
    """Test the conceptual end-to-end P2 flow with synthetic data.

    This simulates the complete pipeline on a tiny 8-document corpus
    to verify that the algorithmic pieces compose correctly:
      similarity construction → fusion → spectral clustering → evaluation
    """

    def test_end_to_end_with_synthetic_data(self):
        """Full pipeline on 8 synthetic documents with 2 clear types.

        Documents 0-3 are "type A" (similar embeddings + similar structure).
        Documents 4-7 are "type B" (different embeddings + different structure).

        With perfect separation, spectral clustering should achieve NMI ≈ 1.0.
        """
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import normalized_mutual_info_score

        n = 8
        gt = [0, 0, 0, 0, 1, 1, 1, 1]

        # Build similarity matrices with clear 2-cluster structure
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if gt[i] == gt[j]:
                    S[i, j] = 0.8 + 0.2 * (i == j)  # high within-cluster
                else:
                    S[i, j] = 0.1  # low between-cluster

        # Spectral clustering with precomputed affinity
        sc = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            random_state=42,
        )
        labels = sc.fit_predict(S)

        # NMI should be perfect (or near-perfect)
        nmi = normalized_mutual_info_score(gt, labels)
        self.assertGreater(nmi, 0.9)

    def test_canonical_fusion_weights_produce_clean_separation(self):
        """The retained 0.5 / 0.3 / 0.2 weights should preserve clean clusters."""
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import normalized_mutual_info_score

        n = 8
        gt = [0, 0, 0, 0, 1, 1, 1, 1]

        # Build three similarity matrices with different noise levels
        base = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                base[i, j] = 0.9 if gt[i] == gt[j] else 0.1

        np.random.seed(42)
        S_sem = np.clip(base + np.random.randn(n, n) * 0.05, 0, 1)
        S_tfidf = np.clip(base + np.random.randn(n, n) * 0.1, 0, 1)
        S_tree = np.clip(base + np.random.randn(n, n) * 0.15, 0, 1)

        # Symmetrize
        for S in [S_sem, S_tfidf, S_tree]:
            S[:] = (S + S.T) / 2
            np.fill_diagonal(S, 1.0)

        S_fused = 0.5 * S_sem + 0.3 * S_tfidf + 0.2 * S_tree
        sc = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            random_state=42,
        )
        labels = sc.fit_predict(S_fused)
        nmi = normalized_mutual_info_score(gt, labels)

        self.assertGreater(nmi, 0.8)


if __name__ == "__main__":
    unittest.main()
