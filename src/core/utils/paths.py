# -*- coding: utf-8 -*-
"""
Path Manager (Refactored for Hybrid Layout)

Manages paths for the LSF pipeline.
Supports a Hybrid Layout to balance Cost vs. Isolation:
- **Shared Assets (Costly)**: Preprocessing, Reconstruction, Embeddings, Labels -> stored in `datasets/`.
- **Experiment Assets (Cheap/Iterative)**: Splits, Models, Results -> stored in `experiments/`.

Directory Layout:
    datasets/{dataset}/latest/          # Shared Assets
        ├── raw/                        # Raw PDFs
        ├── processing/                 # [NEW] Preprocess/Reconstruct Output JSONs
        ├── embedding/                  # Embeddings
        ├── label/                      # Labels
        ├── ground_truth/               # GT Answers
        └── queries.txt                 # Questions

    experiments/{experiment}/{dataset}/ # Experiment Artifacts
        ├── splits[_mineru]/            # Train/Test Splits
        ├── models/                     # Trained Models
        └── results/                    # Evaluation Metrics & Results

"""

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PathManager:
    """
    Unified Path Manager for [Problem 1] Standard Process.
    """

    # Default relative paths within the Dataset Root
    DEFAULT_DATASET_PATHS = {
        "raw": "raw",
        "intermediate": "processing",  # intermediate and processing share the same directory
        "processing": "processing",
        "embedding": "embedding",
        "label": "label",
        "ground_truth": "ground_truth",
        "queries": "queries.txt",
        "knowledge_base": "style_pattern_kb.json",
    }

    def __init__(
        self,
        experiment: str = "default",
        project_root: Optional[Path] = None,
        processing_variant: Optional[str] = None,
    ):
        self.experiment = experiment
        self.processing_variant = processing_variant  # None=docling, "mineru"=mineru
        if project_root is None:
            # core is under src/core/, so project root is the parent of src/
            self.project_root = Path(__file__).resolve().parents[3]
        else:
            self.project_root = Path(project_root)
        self._validated_datasets: set[str] = set()

    @property
    def reconstructed_tag(self) -> str:
        """Suffix tag for model/results directory names, e.g., 'reconstructed' or 'reconstructed_mineru'."""
        if self.processing_variant:
            return f"reconstructed_{self.processing_variant}"
        return "reconstructed"

    @property
    def variant_suffix(self) -> str:
        """Directory-level suffix, e.g., '' or '_mineru'."""
        if self.processing_variant:
            return f"_{self.processing_variant}"
        return ""

    def _get_dataset_path(self, dataset: str, key: str) -> Path:
        """Helper to resolve a path within the dataset root."""
        self._assert_path_yaml_removed(dataset)
        rel_path = self.DEFAULT_DATASET_PATHS.get(key, key)
        return self.project_root / "datasets" / dataset / "latest" / rel_path

    def _assert_path_yaml_removed(self, dataset: str) -> None:
        """`path.yaml` is deprecated; fail immediately if detected to avoid silently using wrong directories."""
        if dataset in self._validated_datasets:
            return

        config_path = self.get_dataset_root(dataset) / "path.yaml"
        if config_path.exists():
            raise RuntimeError(
                "Detected deprecated dataset path override: "
                f"{config_path}. `path.yaml` is no longer supported. "
                "Please migrate the dataset back to the standard directory layout "
                "under `datasets/<dataset>/latest/` and remove `path.yaml`."
            )
        self._validated_datasets.add(dataset)

    # =========================================================================
    # Shared Assets (Cost-intensive, reusable)
    # =========================================================================

    def get_dataset_root(self, dataset: str) -> Path:
        return self.project_root / "datasets" / dataset / "latest"

    def get_data_dir(self, dataset: str) -> Path:
        return self._get_dataset_path(dataset, "raw")

    def get_intermediate_dir(self, dataset: str) -> Path:
        return self._get_dataset_path(dataset, "intermediate")

    def get_processing_dir(self, dataset: str) -> Path:
        base = self._get_dataset_path(dataset, "processing")
        if self.processing_variant:
            return base.with_name(f"{base.name}{self.variant_suffix}")
        return base

    def get_embeddings_dir(self, dataset: str, provider: str = "openai") -> Path:
        base = self._get_dataset_path(dataset, "embedding")
        return base / provider / f"document_embedding{self.variant_suffix}"

    def get_tree_embeddings_dir(self, dataset: str) -> Path:
        """Tree-LSTM cache directory, isolated by parser suffix."""
        base = self._get_dataset_path(dataset, "embedding")
        return base / f"tree_lstm{self.variant_suffix}"

    def get_labels_dir(self, dataset: str) -> Path:
        base = self._get_dataset_path(dataset, "label")
        if self.processing_variant:
            return base.with_name(f"{base.name}{self.variant_suffix}")
        return base

    def get_logs_dir(self, dataset: str) -> Path:
        """Get the label logs directory."""
        return self.get_labels_dir(dataset) / "logs"

    def get_ground_truth_dir(self, dataset: str) -> Path:
        return self._get_dataset_path(dataset, "ground_truth")

    def get_questions_path(self, dataset: str) -> Path:
        return self._get_dataset_path(dataset, "queries")

    def get_knowledge_base_path(self, dataset: str) -> Path:
        return self._get_dataset_path(dataset, "knowledge_base")

    # =========================================================================
    # Experiment Artifacts (Iterative, isolated)
    # =========================================================================

    def get_experiment_root(self, dataset: str) -> Path:
        return self.project_root / "experiments" / self.experiment / dataset

    def get_splits_dir(self, dataset: str) -> Path:
        base = self.get_experiment_root(dataset) / "splits"
        if self.processing_variant:
            return base.with_name(f"{base.name}{self.variant_suffix}")
        return base

    def get_models_dir(self, dataset: str) -> Path:
        return self.get_experiment_root(dataset) / "models"

    def get_results_dir(self, dataset: str) -> Path:
        return self.get_experiment_root(dataset) / "results"

    # =========================================================================
    # Helpers
    # =========================================================================

    def get_reconstructed_json_path(self, dataset: str, doc_name: str) -> Path:
        """Reconstructed JSON is stored in the processing directory."""
        return self.get_processing_dir(dataset) / f"{doc_name}_reconstructed.json"

    def get_processing_json_path(self, dataset: str, doc_name: str) -> Path:
        """Get the standard processed JSON path (Problem 1 pipeline)."""
        return self.get_reconstructed_json_path(dataset, doc_name)

    def get_query_embedding_path(self, dataset: str, provider: str = "openai") -> Path:
        # Cache query embeddings in the dataset dir (reusable)
        base = self._get_dataset_path(dataset, "embedding")
        return base / provider / "query_embeddings.json"
