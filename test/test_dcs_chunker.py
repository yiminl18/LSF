"""Unit tests for DCS adaptive chunker."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np

from agent.baselines.dcs.chunker import chunk, Chunk, _split_sentences


FIXTURE_TEXT = (
    "Amazon was incorporated in 1994 by Jeff Bezos. "
    "The company started as an online bookstore. "
    "Over the years it expanded into cloud computing. "
    "AWS is now a major revenue driver for Amazon. "
    "The company is headquartered in Seattle, Washington. "
    "Amazon operates in North America, Europe, and internationally. "
    "Jeff Bezos stepped down as CEO in 2021. "
    "Andy Jassy became the new Chief Executive Officer. "
    "The company reported record revenues in fiscal year 2022. "
    "E-commerce and AWS both contributed significantly to growth."
)


class FakeEmbedder:
    """Stub SentenceTransformer that returns deterministic embeddings."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(sentences), 16)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-9)


class TestSplitSentences(unittest.TestCase):
    def test_basic_split(self) -> None:
        text = "Hello world. How are you? I'm fine!"
        sents = _split_sentences(text)
        self.assertGreaterEqual(len(sents), 2)
        for s in sents:
            self.assertTrue(s.strip(), "empty sentence in output")

    def test_empty(self) -> None:
        self.assertEqual(_split_sentences(""), [])

    def test_single_sentence(self) -> None:
        sents = _split_sentences("Just one sentence")
        self.assertEqual(len(sents), 1)


class TestChunker(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeEmbedder()

    def test_returns_chunks(self) -> None:
        chunks = chunk(FIXTURE_TEXT, doc_id="test_doc", model=self.model)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIsInstance(c, Chunk)
            self.assertTrue(c.text.strip(), "empty chunk text")

    def test_no_overlap(self) -> None:
        """Sentence indices must be monotonically increasing and non-overlapping."""
        chunks = chunk(FIXTURE_TEXT, doc_id="test_doc", model=self.model)
        all_indices: list[int] = []
        for c in chunks:
            all_indices.extend(c.sentence_indices)
        # All indices should be unique
        self.assertEqual(len(all_indices), len(set(all_indices)), "duplicate sentence indices")
        # Should be sorted across chunk order
        self.assertEqual(all_indices, sorted(all_indices), "sentence indices not monotonic")

    def test_covers_all_sentences(self) -> None:
        """Every sentence from the input should appear in exactly one chunk."""
        from agent.baselines.dcs.chunker import _split_sentences
        sentences = _split_sentences(FIXTURE_TEXT)
        chunks = chunk(FIXTURE_TEXT, doc_id="test_doc", model=self.model)
        all_indices: list[int] = []
        for c in chunks:
            all_indices.extend(c.sentence_indices)
        self.assertEqual(sorted(all_indices), list(range(len(sentences))))

    def test_single_sentence_text(self) -> None:
        result = chunk("Just one sentence.", doc_id="single", model=self.model)
        self.assertEqual(len(result), 1)
        self.assertIn("Just one sentence", result[0].text)

    def test_empty_text(self) -> None:
        result = chunk("", doc_id="empty", model=self.model)
        self.assertEqual(result, [])

    def test_average_chunk_size_in_range(self) -> None:
        """Average chunk length should be at least 100 chars for typical text."""
        chunks = chunk(FIXTURE_TEXT, doc_id="test_doc", model=self.model)
        if len(chunks) > 1:
            avg = sum(len(c.text) for c in chunks) / len(chunks)
            self.assertGreater(avg, 50, f"average chunk size {avg:.0f} chars too small")


if __name__ == "__main__":
    unittest.main()
