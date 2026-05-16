from __future__ import annotations

import json

import pytest

from agent.rule_runtime.data import get_query_text


def test_get_query_text_falls_back_to_queries_json(tmp_path):
    root = tmp_path / "dataset" / "latest"
    root.mkdir(parents=True)
    (root / "queries.json").write_text(
        json.dumps(
            [
                {"text": "First question?", "answer_type": "string"},
                {"text": "Second question?", "answer_type": "date"},
            ]
        ),
        encoding="utf-8",
    )

    assert get_query_text(root, 1) == "Second question?"


def test_get_query_text_prefers_queries_txt(tmp_path):
    root = tmp_path / "dataset" / "latest"
    root.mkdir(parents=True)
    (root / "queries.txt").write_text("Text query\n", encoding="utf-8")
    (root / "queries.json").write_text(
        json.dumps([{"text": "JSON query"}]),
        encoding="utf-8",
    )

    assert get_query_text(root, 0) == "Text query"


def test_get_query_text_errors_on_bad_queries_json_entry(tmp_path):
    root = tmp_path / "dataset" / "latest"
    root.mkdir(parents=True)
    (root / "queries.json").write_text(json.dumps([{"answer_type": "string"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="text"):
        get_query_text(root, 0)
