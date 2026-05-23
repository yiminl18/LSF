"""Compute partial QA metrics for an output directory.

Used for periodic progress snapshots while a run is still in progress.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_token_count(text_dir: Path, doc_name: str) -> float:
    txt = (text_dir / f"{doc_name}.txt").read_text(encoding="utf-8", errors="replace")
    return max(len(txt) / 4.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--queries-file", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    text_dir = Path(args.text_dir)
    queries = _load_json(Path(args.queries_file))
    run_metadata = _load_json(output_dir / "run_metadata.json")
    selected_docs = run_metadata.get("selected_docs") or []
    total_pairs = len(selected_docs) * len(queries)

    records = []
    for question_dir in output_dir.iterdir():
        if not question_dir.is_dir() or question_dir.name == "logs":
            continue
        for path in question_dir.glob("*.json"):
            try:
                records.append(_load_json(path))
            except Exception:
                continue

    completed = len(records)
    n_correct = sum(1 for r in records if r.get("correct") is True)
    accuracy = (n_correct / completed) if completed else 0.0
    avg_latency = (
        sum(float(r.get("latency_seconds") or 0.0) for r in records) / completed if completed else 0.0
    )
    avg_cost_ratio = (
        sum((float(r.get("input_tokens") or 0.0) / _doc_token_count(text_dir, r["doc_name"])) for r in records)
        / completed
        if completed
        else 0.0
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_pairs": completed,
        "total_pairs": total_pairs,
        "progress": round((completed / total_pairs), 4) if total_pairs else 0.0,
        "accuracy": round(accuracy, 4),
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_cost_ratio": round(avg_cost_ratio, 4),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
