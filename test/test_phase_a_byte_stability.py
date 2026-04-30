from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path("test/fixtures/phase_a_byte_stability_baseline.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_phase_a_byte_stability_existing_experiments() -> None:
    """Gate the Wave 0 artifacts from the six existing Phase A commands.

    The gate covers best_rules.json and phase_a_docs.json only. It deliberately
    excludes phase_a_report.md and phase_a_report.json because their manifest
    timestamps are nondeterministic.
    """
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    missing = [path for path in expected if not Path(path).exists()]
    if missing:
        pytest.skip(
            "Wave 0 baseline artifacts are not present; run the six pinned "
            "Phase A commands before using this local hash gate."
        )

    actual = {path: _sha256(Path(path)) for path in expected}

    assert actual == expected
