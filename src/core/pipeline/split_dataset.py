#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split Dataset Module

Splits the dataset into training and testing sets by document ID.
Reads labels from SHARED datasets directory.
Writes split manifests to EXPERIMENT directory.

Usage:
    python -m core.pipeline.split_dataset --dataset pdfs --ratio 0.7 --parser docling
"""

import argparse
import json
import random
import sys

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))


def split_dataset(
    dataset: str = "pdfs",
    ratio: float = 0.7,
    seed: int = 42,
    experiment: str = "default",
    include_none_gt: bool = False,
    label_tag: str = "",
    parser: str = "docling",
    max_train: int | None = None,
) -> None:
    """
    Split dataset into train/test sets (per-question independent).

    Each question is split independently into train/test (no shared doc split).
    Labels without provenance are excluded.

    Args:
        dataset: Dataset name.
        ratio: Train ratio.
        seed: Random seed.
        experiment: Experiment ID (used for output).
        include_none_gt: Include 'None' ground truth labels.
        label_tag: Only use label files with this tag (e.g., 10k_q0_<tag>_reconstructed_labels.json).
        parser: Structural parser (docling/mineru).
    """
    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)

    # Read from SHARED
    labels_dir = paths.get_labels_dir(dataset)
    # Write to EXPERIMENT
    split_dir = paths.get_splits_dir(dataset)
    split_dir.mkdir(parents=True, exist_ok=True)

    print("=== Split Dataset (Per-Question Independent) [Problem 1 Step 5] ===")
    print(f"Dataset:    {dataset}")
    print(f"Experiment: {experiment}")
    print(f"Labels Dir: {labels_dir} (SHARED)")
    print(f"Split Dir:  {split_dir} (EXPERIMENT)")
    print(f"Ratio:      {ratio}")
    print(f"Max Train:  {max_train or '(unlimited)'}")
    print(f"Seed:       {seed}")
    print(f"Include None GT: {include_none_gt}")
    print(f"Label Tag:  {label_tag or '(default)'}")
    print()

    # Per-question independent split
    _split_labels_by_question(
        paths,
        dataset,
        ratio,
        seed,
        include_none_gt,
        label_tag=label_tag,
        max_train=max_train,
    )


def _split_labels_by_question(
    paths: PathManager,
    dataset: str,
    ratio: float,
    seed: int,
    include_none_gt: bool = False,
    label_tag: str = "",
    max_train: int | None = None,
) -> None:
    """
    Generate per-question independent splits.

    Each question is split independently into train/test (no shared doc split).
    Labels without provenance are excluded.
    """
    # Read SHARED labels
    labels_dir = paths.get_labels_dir(dataset)
    # Write EXPERIMENT splits
    split_dir = paths.get_splits_dir(dataset)
    split_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_label(label: dict) -> bool:
        # Exclude 'None' ground truth
        if not include_none_gt and label.get("ground_truth") == "None":
            return False
        # Exclude labels without provenance
        return bool(label.get("possible_provenance_nodes"))

    total_stats = {"train": 0, "test": 0, "valid": 0, "skipped_no_prov": 0}

    # generate_labels output format: {prefix}_q{idx}[_{tag}]_reconstructed_labels.json
    from core.config import get_label_prefix

    prefix = get_label_prefix(dataset)
    if label_tag:
        file_glob = f"{prefix}_q*_{label_tag}_reconstructed_labels.json"
    else:
        file_glob = f"{prefix}_q*_reconstructed_labels.json"
    for labels_file in sorted(labels_dir.glob(file_glob)):
        if "_train_" in labels_file.name or "_test_" in labels_file.name:
            continue
        if not label_tag:
            # When no tag: exact match 10k_q{idx}_{variant}_labels (4 parts)
            # Exclude tagged files like 10k_q0_ablation_merged_labels (5+ parts)
            parts = labels_file.stem.split("_")
            if len(parts) != 4:
                continue

        q_idx = labels_file.stem.split("_")[1]  # e.g., "q0"

        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_labels = data.get("labels", [])

        # Count labels without provenance
        no_prov_count = sum(
            1 for l in all_labels if not l.get("possible_provenance_nodes")
        )
        total_stats["skipped_no_prov"] += no_prov_count

        # Filter valid labels
        valid_labels = [l for l in all_labels if is_valid_label(l)]

        if not valid_labels:
            print(f"  {q_idx}: No valid labels, skipping")
            continue

        # Split independently by doc_name
        doc_names = sorted(set(l["doc_name"] for l in valid_labels))

        random.seed(seed)
        random.shuffle(doc_names)

        split_idx = int(len(doc_names) * ratio)
        if max_train is not None and split_idx > max_train:
            split_idx = max_train
        if len(doc_names) >= 2 and split_idx < 1:
            split_idx = 1
        if len(doc_names) == 1:
            print(f"  {q_idx}: Warning: only 1 document, assigning to test only")
        train_docs = set(doc_names[:split_idx])
        test_docs = set(doc_names[split_idx:])

        train_labels = [l for l in valid_labels if l["doc_name"] in train_docs]
        test_labels = [l for l in valid_labels if l["doc_name"] in test_docs]

        # Update statistics
        total_stats["train"] += len(train_labels)
        total_stats["test"] += len(test_labels)
        total_stats["valid"] += len(valid_labels)

        # Save to EXPERIMENT split directory
        train_path = split_dir / f"{q_idx}_train_labels.json"
        test_path = split_dir / f"{q_idx}_test_labels.json"

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump({"total": len(train_labels), "labels": train_labels}, f, indent=2)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump({"total": len(test_labels), "labels": test_labels}, f, indent=2)

        print(
            f"  {q_idx}: {len(train_labels)} train, {len(test_labels)} test "
            f"(from {len(valid_labels)} valid, {len(doc_names)} docs)"
        )

    print(f"\nTotal: {total_stats['train']} train, {total_stats['test']} test")
    print(
        f"Valid labels: {total_stats['valid']}, Skipped (no provenance): {total_stats['skipped_no_prov']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split Dataset (Per-Question Independent)"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--experiment", type=str, default="default", help="Experiment ID"
    )
    parser.add_argument(
        "--include-none-gt", action="store_true", help="Include 'None' GT labels"
    )
    parser.add_argument(
        "--label-tag", type=str, default="", help="Only read label files with this tag"
    )
    parser.add_argument(
        "--max-train", type=int, default=None, help="Max training documents per query"
    )
    parser.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        dest="struct_parser",
        help="Structural parser (docling/mineru)",
    )

    args = parser.parse_args()

    split_dataset(
        dataset=args.dataset,
        ratio=args.ratio,
        seed=args.seed,
        experiment=args.experiment,
        include_none_gt=args.include_none_gt,
        label_tag=args.label_tag,
        parser=args.struct_parser,
        max_train=args.max_train,
    )


if __name__ == "__main__":
    main()
