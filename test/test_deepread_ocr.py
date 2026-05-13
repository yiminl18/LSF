"""Unit tests for DeepRead OCR module and ParagraphIndex."""

from __future__ import annotations

import unittest

from agent.baselines.deepread.index import (
    ParagraphIndex,
    ParagraphCoord,
    Paragraph,
    SectionMeta,
)
from agent.baselines.deepread.tools import retrieve, read_section
from agent.baselines.deepread.ocr import _parse_page_ocr


class TestParagraphIndex(unittest.TestCase):
    def _make_index(self) -> ParagraphIndex:
        """Build a small fixture index."""
        data = {
            "sections": [
                {
                    "section_id": 1,
                    "heading": "Overview",
                    "level": 1,
                    "page_no": 1,
                    "paragraphs": [
                        {"text": "Amazon is a technology company.", "page_no": 1},
                        {"text": "It operates globally.", "page_no": 1},
                    ],
                },
                {
                    "section_id": 2,
                    "heading": "Products",
                    "level": 2,
                    "page_no": 2,
                    "paragraphs": [
                        {"text": "AWS is a cloud platform.", "page_no": 2},
                        {"text": "Prime is a subscription service.", "page_no": 2},
                        {"text": "Alexa is a voice assistant.", "page_no": 2},
                    ],
                },
            ]
        }
        return ParagraphIndex.from_ocr_result(data)

    def test_from_ocr_result_basic(self) -> None:
        idx = self._make_index()
        self.assertEqual(len(idx.paragraphs), 5)
        self.assertEqual(len(idx.sections), 2)

    def test_monotonic_section_ids(self) -> None:
        idx = self._make_index()
        sids = sorted(idx.sections.keys())
        self.assertEqual(sids, list(range(1, len(sids) + 1)))

    def test_contiguous_in_section_order(self) -> None:
        idx = self._make_index()
        for sid in idx.sections:
            paras = sorted(
                [p for p in idx.paragraphs if p.coord.section_id == sid],
                key=lambda p: p.coord.in_section_order,
            )
            orders = [p.coord.in_section_order for p in paras]
            self.assertEqual(orders, list(range(len(paras))), f"section {sid} not contiguous")

    def test_retrieve_returns_coords(self) -> None:
        idx = self._make_index()
        hits = retrieve(idx, "cloud platform", k=2)
        self.assertIsInstance(hits, list)
        self.assertGreater(len(hits), 0)
        for hit in hits:
            self.assertIsInstance(hit.coord, ParagraphCoord)
            self.assertIsInstance(hit.snippet, str)

    def test_retrieve_empty_query(self) -> None:
        idx = self._make_index()
        hits = retrieve(idx, "", k=3)
        # empty query → no results expected or at least no crash
        self.assertIsInstance(hits, list)

    def test_read_section_full(self) -> None:
        idx = self._make_index()
        text = read_section(idx, section_id=2)
        self.assertIn("AWS", text)
        self.assertIn("Prime", text)
        self.assertIn("Alexa", text)

    def test_read_section_clamped(self) -> None:
        idx = self._make_index()
        text = read_section(idx, section_id=2, start=0, end=1)
        self.assertIn("AWS", text)
        # Second paragraph should not appear
        self.assertNotIn("Prime", text)

    def test_read_section_max_chars(self) -> None:
        idx = self._make_index()
        # Call the index method directly (tools wrapper hardcodes 4000 chars limit)
        text = idx.read_section(section_id=2, max_chars=10)
        self.assertLessEqual(len(text), 10)

    def test_read_section_out_of_bounds(self) -> None:
        idx = self._make_index()
        # section 99 doesn't exist — should return empty string, not raise
        text = read_section(idx, section_id=99)
        self.assertEqual(text, "")

    def test_to_dict_roundtrip(self) -> None:
        idx = self._make_index()
        d = idx.to_dict()
        idx2 = ParagraphIndex.from_ocr_result(d)
        self.assertEqual(len(idx.paragraphs), len(idx2.paragraphs))
        self.assertEqual(len(idx.sections), len(idx2.sections))


class TestParsePageOcr(unittest.TestCase):
    def test_basic_parse(self) -> None:
        raw = (
            "# Overview\n"
            '<p sid="1" pid="1">Amazon is a company.</p>\n'
            '<p sid="1" pid="2">It sells goods online.</p>\n'
            "## Products\n"
            '<p sid="2" pid="1">AWS is their cloud offering.</p>\n'
        )
        counter = [0]
        sections = _parse_page_ocr(page_no=1, raw=raw, global_section_counter=counter)
        self.assertIsInstance(sections, list)
        # There should be at least one section with paragraphs
        total_paras = sum(len(s["paragraphs"]) for s in sections)
        self.assertGreater(total_paras, 0)

    def test_monotonic_section_ids(self) -> None:
        raw = (
            '<p sid="1" pid="1">First para.</p>\n'
            '<p sid="2" pid="1">Second section para.</p>\n'
        )
        counter = [0]
        sections = _parse_page_ocr(page_no=1, raw=raw, global_section_counter=counter)
        sids = [s["section_id"] for s in sections]
        self.assertEqual(sids, sorted(sids))

    def test_global_counter_increments(self) -> None:
        counter = [5]  # start from 5
        raw = '<p sid="1" pid="1">Para.</p>\n'
        sections = _parse_page_ocr(page_no=1, raw=raw, global_section_counter=counter)
        if sections:
            self.assertEqual(sections[0]["section_id"], 6)

    def test_empty_page(self) -> None:
        counter = [0]
        sections = _parse_page_ocr(page_no=1, raw="", global_section_counter=counter)
        self.assertEqual(sections, [])


if __name__ == "__main__":
    unittest.main()
