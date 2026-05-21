"""Convert nopv ground_truth/*.txt_answers.json into the legacy label schema.

The baselines stack (`agent.rule_runtime.data.extract_ground_truth`) reads
`label/10k_q{idx}_reconstructed_labels.json` shaped as::

    {"labels": [{"doc_name": "...", "question_idx": <0-based>, "ground_truth": "..."}]}

`nopv` ships GT in a per-doc dict keyed by 1-based query indices with mixed
value types (str / int / bool / list). This script rewrites it into the
per-query legacy schema, stringifying non-string values via
``normalize_ground_truth`` so the downstream judge/scorer never sees a
non-str ``ground_truth``.

Usage::

    PYTHONPATH=src python3 scripts/convert_nopv_gt_to_labels.py \
        --dataset-root datasets/nopv/latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from core.pipeline.e2e_utils.judge import normalize_ground_truth


def _doc_id_from_gt_filename(name: str) -> str:
    """Strip the `.txt_answers.json` suffix to recover the PDF stem.

    `<stem>.pdf` lives under `raw/`; the GT file is `<stem>.txt_answers.json`.
    """
    if not name.endswith(".txt_answers.json"):
        raise ValueError(f"Unexpected GT filename: {name!r}")
    return name[: -len(".txt_answers.json")]


def _load_queries(queries_path: Path) -> list[dict[str, Any]]:
    with queries_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{queries_path} must contain a JSON list")
    return data


def convert(dataset_root: Path) -> dict[str, int]:
    """Convert per-doc GT files into per-query legacy label files.

    Returns a stats dict: {"n_queries", "n_docs", "n_labels_written"}.
    """
    gt_dir = dataset_root / "ground_truth"
    label_dir = dataset_root / "label"
    queries_path = dataset_root / "queries.json"

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ground_truth dir not found: {gt_dir}")
    if not queries_path.is_file():
        raise FileNotFoundError(f"queries.json not found: {queries_path}")

    queries = _load_queries(queries_path)
    n_queries = len(queries)

    label_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-(doc, query) GT, indexed by query_idx (0-based).
    per_query: dict[int, list[dict[str, Any]]] = {q: [] for q in range(n_queries)}
    doc_count = 0
    for gt_file in sorted(gt_dir.glob("*.txt_answers.json")):
        doc_id = _doc_id_from_gt_filename(gt_file.name)
        with gt_file.open("r", encoding="utf-8") as f:
            doc_gt = json.load(f)
        if not isinstance(doc_gt, dict):
            raise ValueError(f"{gt_file} must contain a JSON object")
        doc_count += 1
        for query_idx in range(n_queries):
            # nopv keys are 1-based strings: "1".."13"
            key = str(query_idx + 1)
            if key not in doc_gt:
                continue
            normalized = normalize_ground_truth(doc_gt[key])
            if not normalized:
                continue
            per_query[query_idx].append(
                {
                    "doc_name": doc_id,
                    "question_idx": query_idx,
                    "ground_truth": normalized,
                }
            )

    n_labels = 0
    for query_idx, labels in per_query.items():
        out_path = label_dir / f"10k_q{query_idx}_reconstructed_labels.json"
        payload = {"labels": labels}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        n_labels += len(labels)

    return {
        "n_queries": n_queries,
        "n_docs": doc_count,
        "n_labels_written": n_labels,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/nopv/latest"),
        help="Dataset root containing ground_truth/ and queries.json",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    stats = convert(args.dataset_root.resolve())
    print(
        "Converted nopv GT -> label schema:\n"
        f"  queries:   {stats['n_queries']}\n"
        f"  docs:      {stats['n_docs']}\n"
        f"  labels:    {stats['n_labels_written']} entries across "
        f"{stats['n_queries']} files\n"
        f"  output:    {args.dataset_root.resolve() / 'label'}"
    )


if __name__ == "__main__":
    main()
