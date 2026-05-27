"""Phase 1: read A*(d) from an existing eval_merge artefact.

The eval_merge file is produced by test/run_eval_merge_sampled.py and has the
structure:
    {
        "accuracy": 0.9,
        "per_doc": [
            {"doc_name": "AMCOR_2019_10K", "correct": true, ...},
            ...
        ],
        ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path


def load_target_docs(eval_merge_path: Path) -> set[str]:
    """Return doc_names that the full rule set solved (A* = 1)."""
    data = json.loads(eval_merge_path.read_text(encoding="utf-8"))
    return {row["doc_name"] for row in data["per_doc"] if row["correct"]}
