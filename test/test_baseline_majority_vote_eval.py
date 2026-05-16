from __future__ import annotations

import json
from pathlib import Path

from agent.baselines.majority_vote_eval import (
    apply_majority_vote,
    sample_pairs,
    summarize,
)


def _write_dataset(root: Path, dataset: str, query_count: int, doc_count: int) -> None:
    dataset_root = root / dataset / "latest"
    raw = dataset_root / "raw"
    raw.mkdir(parents=True)
    queries = [{"text": f"Question {idx}?"} for idx in range(query_count)]
    (dataset_root / "queries.json").write_text(
        json.dumps(queries),
        encoding="utf-8",
    )
    for idx in range(doc_count):
        (raw / f"doc_{idx}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")


def test_sample_pairs_selects_requested_shape(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "toy", query_count=8, doc_count=10)

    pairs = sample_pairs(
        datasets=["toy"],
        data_root=tmp_path,
        queries_per_dataset=5,
        docs_per_query=5,
        seed=7,
    )

    assert len(pairs) == 25
    assert len({pair.query_idx for pair in pairs}) == 5
    by_query = {}
    for pair in pairs:
        by_query.setdefault(pair.query_idx, set()).add(pair.doc_id)
    assert all(len(doc_ids) == 5 for doc_ids in by_query.values())


def test_sample_pairs_accepts_explicit_query_indices(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "toy", query_count=8, doc_count=10)

    pairs = sample_pairs(
        datasets=["toy"],
        data_root=tmp_path,
        queries_per_dataset=5,
        docs_per_query=2,
        seed=7,
        query_indices=[0, 3],
    )

    assert len(pairs) == 4
    assert {pair.query_idx for pair in pairs} == {0, 3}


def test_majority_vote_marks_matching_answers_correct() -> None:
    rows = [
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "Q?",
            "doc_id": "doc",
            "baseline": "exit",
            "status": "ok",
            "answer": "Amazon",
            "normalized_answer": "amazon",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "Q?",
            "doc_id": "doc",
            "baseline": "deepread",
            "status": "ok",
            "answer": "Amazon.",
            "normalized_answer": "amazon",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "Q?",
            "doc_id": "doc",
            "baseline": "mdocagent",
            "status": "ok",
            "answer": "Google",
            "normalized_answer": "google",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "Q?",
            "doc_id": "doc",
            "baseline": "qa-agent",
            "status": "ok",
            "answer": "Amazon",
            "normalized_answer": "amazon",
            "cost_usd": 0.0,
        },
    ]

    rows, pair_rows = apply_majority_vote(rows)

    assert pair_rows[0]["majority_resolved"] is True
    assert pair_rows[0]["majority_count"] == 3
    assert {
        row["baseline"]: row["majority_vote_correct"]
        for row in rows
    } == {
        "exit": True,
        "deepread": True,
        "mdocagent": False,
        "qa-agent": True,
    }
    summary = summarize(rows, pair_rows)
    assert summary["resolved_pairs"] == 1
    assert summary["baselines"]["exit"]["majority_vote_accuracy"] == 1.0


def test_majority_vote_leaves_two_two_tie_unresolved() -> None:
    rows = []
    for baseline, answer in [
        ("exit", "A"),
        ("deepread", "A"),
        ("mdocagent", "B"),
        ("qa-agent", "B"),
    ]:
        rows.append(
            {
                "dataset": "toy",
                "query_idx": 0,
                "query_text": "Q?",
                "doc_id": "doc",
                "baseline": baseline,
                "status": "ok",
                "answer": answer,
                "normalized_answer": answer.casefold(),
                "cost_usd": 0.0,
            }
        )

    rows, pair_rows = apply_majority_vote(rows)

    assert pair_rows[0]["majority_resolved"] is False
    assert pair_rows[0]["reason"] == "tie"
    assert all(row["majority_vote_correct"] is None for row in rows)


def test_majority_vote_can_cluster_equivalent_answers_and_ignore_non_answers() -> None:
    rows = [
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "When was it issued?",
            "doc_id": "doc",
            "baseline": "exit",
            "status": "ok",
            "answer": "August 29, 2024",
            "normalized_answer": "august 29, 2024",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "When was it issued?",
            "doc_id": "doc",
            "baseline": "deepread",
            "status": "ok",
            "answer": "Information not found.",
            "normalized_answer": "information not found",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "When was it issued?",
            "doc_id": "doc",
            "baseline": "mdocagent",
            "status": "ok",
            "answer": "The notice was issued on August 29, 2024.",
            "normalized_answer": "the notice was issued on august 29, 2024",
            "cost_usd": 0.0,
        },
        {
            "dataset": "toy",
            "query_idx": 0,
            "query_text": "When was it issued?",
            "doc_id": "doc",
            "baseline": "qa-agent",
            "status": "ok",
            "answer": "August 29, 2024",
            "normalized_answer": "august 29, 2024",
            "cost_usd": 0.0,
        },
    ]

    def judge(_question: str, a: str, b: str) -> bool:
        return "August 29, 2024" in a and "August 29, 2024" in b

    rows, pair_rows = apply_majority_vote(rows, equivalence_judge=judge)

    assert pair_rows[0]["majority_resolved"] is True
    assert pair_rows[0]["majority_count"] == 3
    assert pair_rows[0]["valid_vote_count"] == 3
    assert {
        row["baseline"]: row["majority_vote_correct"]
        for row in rows
    } == {
        "exit": True,
        "deepread": False,
        "mdocagent": True,
        "qa-agent": True,
    }
