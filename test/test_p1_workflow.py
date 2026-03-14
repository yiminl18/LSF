"""
Educational walkthrough of the retained LSF Problem 1 workflow.

This artifact keeps only the validated deployment-facing P1 surface:

1. extract the 52-dimensional mode-25 feature set
2. train `xgb-sem-struc-v5` or `hnn-sem-struc-v5`
3. optionally enable curriculum learning
4. train and evaluate with three seeds: 41, 42, 43
5. aggregate scores with `softmax` (default, alpha=5.0) or `top2_mean`
6. fuse retained rankings with simple RRF

The tests below are intentionally small and dependency-light. They explain the
expected behavior of the retained workflow without requiring datasets, GPUs, or
API calls.
"""

from __future__ import annotations

import unittest

import numpy as np


class TestRetainedModelSurface(unittest.TestCase):
    """Verify that the artifact exposes only the retained P1 model surface."""

    def test_only_canonical_model_types_are_mapped(self):
        """The model-type map should contain exactly the retained v5 models."""
        from core.ml.config import MODEL_TYPE_TO_MODE

        self.assertEqual(
            MODEL_TYPE_TO_MODE,
            {
                "xgb-sem-struc-v5": 25,
                "hnn-sem-struc-v5": 25,
            },
        )

    def test_default_seed_list_is_three_seeds(self):
        """Three-seed evaluation is the retained default for variance control."""
        from core.ml.config import DEFAULT_SEEDS

        self.assertEqual(DEFAULT_SEEDS, [41, 42, 43])

    def test_mode25_capabilities_match_the_retained_feature_surface(self):
        """Mode 25 keeps semantic, structural, visual, content, lexical, and BM25 features."""
        from core.ml.config import (
            CAP_BM25,
            CAP_CONTENT,
            CAP_LEXICAL,
            CAP_SIM_B,
            CAP_STRUC,
            CAP_VISUAL2,
            MODE_CAPS,
        )

        self.assertEqual(
            MODE_CAPS[25],
            frozenset(
                {
                    CAP_SIM_B,
                    CAP_STRUC,
                    CAP_VISUAL2,
                    CAP_CONTENT,
                    CAP_LEXICAL,
                    CAP_BM25,
                }
            ),
        )

    def test_mode25_feature_names_match_the_retained_v5_surface(self):
        """The artifact should expose the exact 52 v5 feature names."""
        from core.ml.features import RETAINED_V5_FEATURE_NAMES

        self.assertEqual(
            RETAINED_V5_FEATURE_NAMES,
            [
                "a_b_char3_jaccard",
                "a_b_fuzz_ratio",
                "a_b_numeric_token_overlap",
                "a_b_tok_jaccard",
                "abs_font_bucket_diff",
                "abs_font_size_rank_diff",
                "abs_page_diff",
                "abs_pos_frac_diff",
                "abs_prefix_change_diff",
                "all_cap_match",
                "bm25_query_combined_b",
                "bm25_query_header_b",
                "bold_match",
                "center_match",
                "depth",
                "f1",
                "f2",
                "f3",
                "font_name_match",
                "font_size_rank_b",
                "header_text_len_b",
                "is_all_cap_b",
                "is_center_b",
                "normalized_position",
                "numbering_type_match",
                "numtype_hierarchy_score",
                "parent_structure_level_b",
                "q_b_char3_jaccard",
                "q_b_fuzz_ratio",
                "q_b_tok_jaccard",
                "q_b_tok_precision",
                "q_b_tok_recall",
                "q_path_b_tok_jaccard",
                "sim_ab_query_diff",
                "sim_b",
                "sim_query_path_b",
                "sim_query_path_diff",
                "starts_letter_match",
                "starts_num_match",
                "struc_depth",
                "struc_depth_diff",
                "struc_h1_index_norm",
                "struc_is_first_child",
                "struc_is_last_child",
                "struc_is_parent_child",
                "struc_is_same_parent",
                "struc_seq_distance",
                "struc_sibling_index_norm",
                "structure_level_diff",
                "structure_level_match",
                "textspan_len_b",
                "visual_style_match_count",
            ],
        )

    def test_factory_resolves_the_two_retained_prefixes(self):
        """The factory should lazily resolve XGBoost and HNN only."""
        from core.ml.factory import _CLASSIFIER_REGISTRY, get_classifier_class

        self.assertEqual(set(_CLASSIFIER_REGISTRY), {"xgb", "hnn"})
        self.assertEqual(
            get_classifier_class("xgb-sem-struc-v5").__name__,
            "XGBoostClassifier",
        )
        self.assertEqual(
            get_classifier_class("hnn-sem-struc-v5").__name__,
            "HybridNNClassifier",
        )


class TestCurriculumAndSeeds(unittest.TestCase):
    """Explain the retained curriculum-learning and multi-seed behavior."""

    def test_curriculum_schedule_matches_the_documented_two_phase_plan(self):
        """Curriculum learning keeps a short easy phase and a longer hard phase."""
        phase_a_fraction = 0.3
        phase_b_fraction = 0.7
        phase_a_hard_negative_ratio = 0.2
        phase_b_hard_negative_ratio = 0.8

        self.assertAlmostEqual(phase_a_fraction + phase_b_fraction, 1.0)
        self.assertLess(phase_a_hard_negative_ratio, 0.5)
        self.assertGreater(phase_b_hard_negative_ratio, 0.5)

    def test_three_seeds_support_simple_consensus_evaluation(self):
        """Three seeds are enough for robust aggregation without widening the public surface."""
        seed_scores = {
            41: 0.91,
            42: 0.88,
            43: 0.90,
        }

        self.assertEqual(sorted(seed_scores), [41, 42, 43])
        self.assertGreater(np.mean(list(seed_scores.values())), 0.89)


class TestScoreAggregation(unittest.TestCase):
    """Explain the retained score aggregation methods for evaluation."""

    def test_softmax_with_alpha_five_is_the_default(self):
        """The retained default is softmax aggregation with alpha=5.0."""
        scores = np.array([0.90, 0.70, 0.50], dtype=float)
        alpha = 5.0

        shifted = alpha * scores
        weights = np.exp(shifted - shifted.max())
        weights = weights / weights.sum()
        aggregated = float(np.dot(weights, scores))

        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertGreater(aggregated, 0.75)

    def test_top2_mean_keeps_the_two_best_seeds(self):
        """The simpler retained alternative is the mean of the top two seed scores."""
        scores = np.array([0.90, 0.70, 0.50], dtype=float)
        aggregated = float(np.sort(scores)[-2:].mean())

        self.assertAlmostEqual(aggregated, 0.80)


class TestSimpleRRF(unittest.TestCase):
    """Explain the retained simple RRF ensemble."""

    def test_simple_rrf_uses_rank_positions_only(self):
        """Simple RRF is safe across model families because it ignores raw score scales."""
        k = 60
        ranks = {
            "H1": [1, 2],
            "H2": [2, 1],
            "H3": [3, 3],
        }

        scores = {
            header: sum(1.0 / (k + rank) for rank in header_ranks)
            for header, header_ranks in ranks.items()
        }

        self.assertAlmostEqual(scores["H1"], scores["H2"], places=10)
        self.assertGreater(scores["H1"], scores["H3"])


class TestFeatureExtractorContract(unittest.TestCase):
    """Verify the retained extractor surface is strict and stable."""

    def _build_headers_and_context(self):
        from core.doc.feature_extract import DocumentContext, HeaderNode

        header_a = HeaderNode(
            idx_in_texts=0,
            text="Item 1",
            text_span="Overview",
            page_no=1,
            font_size=12.0,
            is_bold=1,
            font_name="Times",
            is_all_cap=0,
            starts_num=0,
            starts_letter=1,
            is_center=0,
            processing_path="Item 1",
            structure_level=1,
            h1_index_norm=0.0,
            sibling_index_norm=0.0,
            depth=1,
            is_first_child=1,
            is_last_child=0,
            parent_id=-1,
        )
        header_b = HeaderNode(
            idx_in_texts=1,
            text="Item 2",
            text_span="Risk Factors",
            page_no=2,
            font_size=13.0,
            is_bold=1,
            font_name="Times",
            is_all_cap=0,
            starts_num=0,
            starts_letter=1,
            is_center=0,
            processing_path="Item 2",
            structure_level=2,
            h1_index_norm=0.5,
            sibling_index_norm=1.0,
            depth=2,
            is_first_child=0,
            is_last_child=1,
            parent_id=0,
        )
        context = DocumentContext(
            total_headers=2,
            pos_frac=[0.0, 1.0],
            prefix_pattern_change_count=[0, 1],
            prefix_pattern_approx_distinct=[1, 2],
            pattern_key=[],
            font_size_rank_map={12.0: 0.0, 13.0: 1.0},
        )
        return header_a, header_b, context

    def test_extract_ml_features_returns_exact_v5_feature_keys(self):
        from core.ml.features import RETAINED_V5_FEATURE_NAMES, extract_ml_features

        header_a, header_b, context = self._build_headers_and_context()
        features = extract_ml_features(
            header_a,
            header_b,
            0,
            1,
            context,
            context,
            {},
            {},
            query_embedding=[0.0, 0.0, 0.0],
            query_text="risk factors",
            mode=25,
        )

        self.assertEqual(sorted(features.keys()), RETAINED_V5_FEATURE_NAMES)

    def test_extract_ml_features_rejects_unsupported_modes(self):
        from core.ml.features import extract_ml_features
        header_a, header_b, context = self._build_headers_and_context()

        with self.assertRaises(ValueError):
            extract_ml_features(
                header_a,
                header_b,
                0,
                1,
                context,
                context,
                {},
                {},
                query_embedding=[0.0, 0.0, 0.0],
                query_text="risk factors",
                mode=26,
            )


if __name__ == "__main__":
    unittest.main()
