#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Model Module

Trains ranking models (XGBoost) using generated provenance labels.
Reads training splits from EXPERIMENT directory.
Reads processing/embeddings from SHARED directory.
Writes trained models to EXPERIMENT directory.

Usage:
    python -m core.pipeline.train_model --dataset pdfs --model_config xgb-15 --limit 10 --parser docling
"""

import argparse
import json
import sys
import os
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from core.utils.paths import PathManager, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.utils.parallel import run_pool_with_progress
from core.ml.config import (
    MODEL_TYPE_TO_MODE,
    ML_MODEL_TYPES,
    DEFAULT_MODEL_TYPES,
    DEFAULT_SEEDS,
    get_classifier_prefix,
)
from core.ml.dataset import (
    SimilarityDataset,
    ProvenanceAnnotation,
    clear_embeddings_cache,
    clear_document_cache,
)
from core.ml.factory import create_classifier
from core.embed.embeddings import get_query_embedding, get_model_name_for_provider
from core.utils.io import load_labels

# Supported embedding providers
EMBED_PROVIDERS = [
    "openai",
    "azure",
    "openrouter",
]

def _resolve_samples(
    ds: SimilarityDataset,
    mode: int,
    seed: int,
    phase: str,
    cache: Dict[tuple, tuple],
) -> tuple:
    """Resolve sample cache for the retained v5-only feature surface."""
    ck = (mode, seed, phase)
    if ck in cache:
        return cache[ck]
    result = ds.generate_samples(seed=seed)
    cache[ck] = result
    return result


def _compute_provenance_confidence(nodes: List[dict]) -> float:
    """Compute confidence weight from annotation metadata.

    Uses rank and similarity to assess annotation reliability:
    - Low rank + high similarity → easy to retrieve, annotation is trustworthy
    - High rank + low similarity → marginal match, annotation may be noisy
    - Null metadata → manual annotation or early version, default high confidence

    Returns a weight in [0.3, 1.0].
    """
    confidences = []
    for node in nodes:
        # Pseudo-label: fixed low weight
        if node.get("pseudo"):
            confidences.append(0.5)
            continue

        rank = node.get("rank")
        sim = node.get("similarity")

        # No metadata (manual annotation) → high confidence
        if rank is None or sim is None:
            confidences.append(1.0)
            continue

        # Rank component: rank=1→1.0, rank=5→0.88, rank=15→0.70, rank=50→0.40
        rank_conf = 1.0 / (1.0 + 0.02 * (rank - 1))

        # Similarity component: normalized against 0.4 baseline
        sim_conf = min(max(sim / 0.4, 0.0), 1.0)

        # Weighted combination: rank weighted higher (low rank = easy to retrieve)
        conf = 0.6 * rank_conf + 0.4 * sim_conf
        confidences.append(conf)

    avg = sum(confidences) / len(confidences)
    # Floor at 0.3: never fully discard any annotation
    return max(avg, 0.3)


def labels_to_annotations(
    labels: List[dict], question: str, confidence_weight: bool = False
) -> List[ProvenanceAnnotation]:
    """Convert label dicts to a list of ProvenanceAnnotation."""
    annotations = []
    for label in labels:
        nodes = label.get("possible_provenance_nodes", [])
        if not nodes:
            continue
        weight = _compute_provenance_confidence(nodes) if confidence_weight else 1.0
        annotations.append(
            ProvenanceAnnotation(
                doc_id=label["doc_name"],
                question=question,
                answer=label.get("ground_truth", ""),
                refined_provenance=nodes,
                weight=weight,
            )
        )
    return annotations


def _build_params(
    prefix: str,
    seed: int,
    optuna_params: Optional[Dict] = None,
    model_type: str = "",
    xgb_device: str = "cpu",
) -> Dict:
    """Build classifier parameters based on prefix."""
    if prefix == "hnn":
        # HNN params (wider network + residual connection)
        base_params = {
            "hidden_dims": (128, 64, 32),
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "loss_type": "focal",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "use_residual": True,
            "seed": seed,
        }
        # FiLM conditional modulation (enabled when model_type contains 'film')
        if "film" in model_type:
            base_params["use_film"] = True
            base_params["film_cond_dim"] = 16
        # Mixture of Experts (enabled when model_type contains 'moe')
        elif "moe" in model_type:
            base_params["use_moe"] = True
            base_params["film_cond_dim"] = 16
            base_params["moe_num_experts"] = 3
    elif prefix == "xgb":
        # XGBoost binary classification params
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "aucpr"],
            "scale_pos_weight": 5,
            "max_depth": 4,
            "eta": 0.1,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
        }
        if xgb_device == "cuda":
            # CUDA + hist tree method
            base_params.update(
                {
                    "device": "cuda",
                    "tree_method": "hist",
                    "predictor": "gpu_predictor",
                }
            )
    else:
        base_params = {"seed": seed}

    if optuna_params:
        base_params.update(optuna_params)

    return base_params


def _train_worker(args):
    """Worker for parallel training."""
    (
        dataset,
        q_idx,
        model_configs,
        seeds,
        use_optuna,
        experiment,
        progress_queue,
        provider,
        hard_neg_ratio,
        xgb_device,
        proximal_neg,
        confidence_weight,
        curriculum,
        label_suffix,
        nn_device,
        skip_existing,
        train_fraction,
        parser,
    ) = args

    try:
        return train_models_for_question(
            dataset=dataset,
            q_idx=q_idx,
            model_configs=model_configs,
            seeds=seeds,
            use_optuna=use_optuna,
            experiment=experiment,
            progress_queue=progress_queue,
            provider=provider,
            force=not skip_existing,
            hard_neg_ratio=hard_neg_ratio,
            xgb_device=xgb_device,
            proximal_neg=proximal_neg,
            confidence_weight=confidence_weight,
            curriculum=curriculum,
            label_suffix=label_suffix,
            nn_device=nn_device,
            train_fraction=train_fraction,
            parser=parser,
        )
    except Exception as e:
        if progress_queue:
            progress_queue.put(("log", q_idx, f"[red]Error: {e}[/red]"))
        import traceback

        traceback.print_exc()
        return {}
    finally:
        # Clean up global caches to prevent memory leak in worker process
        clear_embeddings_cache()
        clear_document_cache()
        gc.collect()


def train_models_for_question(
    dataset: str,
    q_idx: int,
    model_configs: List[str],
    seeds: List[int],
    use_optuna: bool,
    experiment: str,
    progress_queue: Optional[Any] = None,
    provider: str = "openrouter",
    force: bool = False,
    hard_neg_ratio: float = 0.6,
    xgb_device: str = "auto",
    proximal_neg: bool = False,
    confidence_weight: bool = False,
    curriculum: bool = False,
    label_suffix: Optional[str] = None,
    nn_device: str = "auto",
    train_fraction: float = 1.0,
    parser: str = "docling",
) -> Dict[str, List[Path]]:
    """
    Train models for a single question.
    """
    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)
    if xgb_device == "auto":
        xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Input: Splits (Experiment)
    splits_dir = paths.get_splits_dir(dataset)
    # Input: Processing/Embeddings (Shared)
    processing_dir = paths.get_processing_dir(dataset)
    embeddings_dir = paths.get_embeddings_dir(dataset, provider=provider)

    models_root = paths.get_models_dir(dataset)
    optuna_dir = models_root / "optuna"

    # Load Train Labels
    suffix = f"_{label_suffix}" if label_suffix else ""
    train_labels_path = splits_dir / f"q{q_idx}_train_labels{suffix}.json"
    train_labels = load_labels(train_labels_path)

    # Document-level subsampling (all annotations for a doc are kept or removed together)
    if train_fraction < 1.0:
        import random as _rng

        doc_names = sorted(set(label["doc_name"] for label in train_labels))
        rng = _rng.Random(42)
        n_keep = max(2, int(len(doc_names) * train_fraction))
        keep = set(rng.sample(doc_names, n_keep))
        train_labels = [label for label in train_labels if label["doc_name"] in keep]

    if not train_labels:
        if progress_queue:
            progress_queue.put(("log", q_idx, "[yellow]No train labels[/yellow]"))
        return {}

    question = train_labels[0].get("question", "")

    annotations = labels_to_annotations(
        train_labels, question, confidence_weight=confidence_weight
    )

    if len(annotations) < 2:
        if progress_queue:
            progress_queue.put(
                (
                    "log",
                    q_idx,
                    f"[yellow]Insufficient positive samples ({len(annotations)})[/yellow]",
                )
            )
        return {}

    # Progress Update
    total_steps = len(model_configs) * len(seeds)
    current_step = 0
    if progress_queue:
        progress_queue.put(("progress", q_idx, 0, total_steps))

    # Pre-compute Query Embedding
    model_name = get_model_name_for_provider(provider)
    query_emb = get_query_embedding(
        question, model=model_name, provider=provider, source=dataset
    )
    query_emb = np.asarray(query_emb, dtype=np.float32).tolist()

    # Load Dataset (Shared Features)
    ds = SimilarityDataset(
        annotations=annotations,
        merged_json_dir=str(processing_dir),
        embeddings_dir=str(embeddings_dir),
        tree_embeddings_dir=str(paths.get_tree_embeddings_dir(dataset)),
        query=question,
        hard_neg_ratio=hard_neg_ratio,
        feature_mode=15,  # Default, will override
        provider=provider,
        proximal_neg=proximal_neg,
        curriculum=curriculum,
        q_idx=q_idx,
    )
    ds.set_query_embedding(query_emb)

    results = {}

    # Samples for the same mode+seed+phase are generated only once when
    # multiple model names share the same feature mode.
    _sample_cache: Dict[tuple, tuple] = {}

    for config in model_configs:
        # Determine mode
        mode = MODEL_TYPE_TO_MODE.get(config)
        if mode is None:
            if progress_queue:
                progress_queue.put(
                    ("log", q_idx, f"[yellow]Unknown mode for {config}[/yellow]")
                )
            current_step += len(seeds)
            if progress_queue:
                progress_queue.put(("progress", q_idx, current_step, total_steps))
            continue

        # Output Model Dir (Experiment)
        config_dir_name = config
        if train_fraction < 1.0:
            pct = int(train_fraction * 100)
            config_dir_name = f"{config}_snap_frac{pct}"
        model_type_dir = models_root / config_dir_name
        model_type_dir.mkdir(parents=True, exist_ok=True)

        provider_dir = (
            model_type_dir / f"{dataset}_{provider}_{paths.reconstructed_tag}"
        )
        provider_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for seed in seeds:
            model_filename = f"q{q_idx}_seed{seed}.json"
            model_path = provider_dir / model_filename

            if model_path.exists() and not force:
                saved_paths.append(model_path)
                current_step += 1
                if progress_queue:
                    progress_queue.put(("progress", q_idx, current_step, total_steps))
                continue

            try:
                ds.feature_mode = mode

                # Optuna params
                optuna_params = None
                if use_optuna:
                    opt_path = optuna_dir / config / f"q{q_idx}_best_params.json"
                    if opt_path.exists():
                        with open(opt_path) as f:
                            optuna_params = json.load(f).get("best_params")

                prefix = get_classifier_prefix(config)
                params = _build_params(
                    prefix,
                    seed,
                    optuna_params,
                    model_type=config,
                    xgb_device=xgb_device,
                )

                if curriculum:
                    # Two-phase curriculum learning:
                    # Phase A (30%): easy negatives (hard_neg_ratio=0.2)
                    # Phase B (70%): hard negatives (hard_neg_ratio=0.8)
                    # For listwise loss, Phase A pre-trains with focal, Phase B switches to listwise
                    total_rounds = 100
                    phase_a_rounds = int(total_rounds * 0.3)
                    phase_b_rounds = total_rounds - phase_a_rounds

                    original_loss_type = params.get("loss_type", "focal")
                    is_listwise_nn = original_loss_type in (
                        "approxndcg",
                        "listmle",
                    ) and prefix in ("hnn", "fca")

                    # Phase A: easy (listwise models pre-train with focal first)
                    params_a = params
                    if is_listwise_nn:
                        params_a = {**params, "loss_type": "focal"}

                    _ck_a = (mode, seed, "easy")
                    if _ck_a not in _sample_cache:
                        ds.curriculum_phase = "easy"
                        _resolve_samples(ds, mode, seed, "easy", _sample_cache)
                    X_a, y_a, feature_names, doc_ids_a, weights_a, _, group_keys_a = (
                        _sample_cache[_ck_a]
                    )
                    clf: Any = create_classifier(
                        config,
                        params=params_a,
                        feature_names=feature_names,
                        device=nn_device,
                    )
                    clf.train(
                        X_a,
                        y_a,
                        doc_ids=doc_ids_a,
                        group_keys=group_keys_a,
                        num_rounds=phase_a_rounds,
                        early_stopping_rounds=None,
                        verbose=False,
                        use_mrr=False,
                        sample_weight=weights_a,
                    )

                    # Phase B: hard, warm-start from Phase A weights
                    # Listwise models: switch back to target loss
                    if is_listwise_nn:
                        clf.params["loss_type"] = original_loss_type

                    _ck_b = (mode, seed, "hard")
                    if _ck_b not in _sample_cache:
                        ds.curriculum_phase = "hard"
                        _resolve_samples(ds, mode, seed, "hard", _sample_cache)
                    X_b, y_b, _, doc_ids_b, weights_b, _, group_keys_b = _sample_cache[
                        _ck_b
                    ]
                    if hasattr(clf, "model") and hasattr(clf.model, "state_dict"):
                        phase_a_state = {
                            k: v.cpu().clone()
                            for k, v in clf.model.state_dict().items()
                        }
                        clf.train(
                            X_b,
                            y_b,
                            doc_ids=doc_ids_b,
                            group_keys=group_keys_b,
                            num_rounds=phase_b_rounds,
                            verbose=False,
                            use_mrr=True,
                            sample_weight=weights_b,
                            init_state_dict=phase_a_state,
                        )
                    else:
                        # XGB: warm-start via xgb_model parameter
                        import xgboost as xgb

                        phase_a_booster = clf.model
                        ds2_feature_names = feature_names
                        dtrain_b = xgb.DMatrix(
                            X_b, label=y_b, feature_names=ds2_feature_names
                        )
                        clf.model = xgb.train(
                            params,
                            dtrain_b,
                            num_boost_round=phase_b_rounds,
                            xgb_model=phase_a_booster,
                            verbose_eval=False,
                        )
                else:
                    # Standard single-phase training
                    _ck = (mode, seed, "standard")
                    if _ck not in _sample_cache:
                        _resolve_samples(ds, mode, seed, "standard", _sample_cache)
                    X, y, feature_names, doc_ids, weights, meta_list, group_keys = (
                        _sample_cache[_ck]
                    )
                    clf: Any = create_classifier(
                        config,
                        params=params,
                        feature_names=feature_names,
                        device=nn_device,
                    )
                    clf.train(
                        X,
                        y,
                        doc_ids=doc_ids,
                        group_keys=group_keys,
                        verbose=False,
                        use_mrr=True,
                        sample_weight=weights,
                    )

                clf.save(str(model_path))

                saved_paths.append(model_path)

            except Exception as e:
                if progress_queue:
                    progress_queue.put(
                        ("log", q_idx, f"[red]Train Fail {config}: {e}[/red]")
                    )
                import traceback

                traceback.print_exc()

            current_step += 1
            if progress_queue:
                progress_queue.put(("progress", q_idx, current_step, total_steps))

        results[config] = saved_paths

    return results


def train_models(
    dataset: str = "pdfs",
    limit: int = 10,  # num_questions
    model_configs: Optional[List[str]] = None,
    experiment: str = "default",
    use_optuna: bool = True,
    workers: Optional[int] = None,
    provider: str = "openrouter",
    seed: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    hard_neg_ratio: float = 0.6,
    xgb_device: str = "auto",
    proximal_neg: bool = False,
    confidence_weight: bool = False,
    curriculum: bool = False,
    label_suffix: Optional[str] = None,
    nn_device: str = "auto",
    skip_existing: bool = False,
    train_fraction: float = 1.0,
    parser: str = "docling",
):
    """
    Batch train models.
    """
    model_configs = model_configs if model_configs else DEFAULT_MODEL_TYPES
    if seeds:
        seeds_list = seeds
    elif seed:
        seeds_list = [seed]
    else:
        seeds_list = DEFAULT_SEEDS
    seeds = seeds_list

    default_workers = min(os.cpu_count() or 1, limit)
    train_workers = workers if workers is not None else default_workers

    print("=== Train Models [Problem 1 Step 6] ===")
    print(f"Dataset:    {dataset}")
    print(f"Experiment: {experiment}")
    print(f"Questions:  {limit}")
    print(f"Models:     {model_configs}")
    print(f"Seeds:      {seeds}")
    print(f"Provider:   {provider}")
    print(f"Workers:    {train_workers}")
    print(f"Hard Neg:   {hard_neg_ratio:.0%}")
    print(f"Proximal:   {proximal_neg}")
    print(f"Conf.W:     {confidence_weight}")
    print(f"Curriculum: {curriculum}")
    if label_suffix:
        print(f"Labels:     *_{label_suffix}.json")
    print(f"XGB Device: {xgb_device}")
    print(f"NN Device:  {nn_device}")
    print(f"Skip Exist: {skip_existing}")
    if train_fraction < 1.0:
        print(f"Fraction:   {train_fraction:.0%}")
    print()

    # Parallel Training
    def _build_train_tasks(progress_queue):
        return [
            (
                dataset,
                q_idx,
                model_configs,
                seeds,
                use_optuna,
                experiment,
                progress_queue,
                provider,
                hard_neg_ratio,
                xgb_device,
                proximal_neg,
                confidence_weight,
                curriculum,
                label_suffix,
                nn_device,
                skip_existing,
                train_fraction,
                parser,
            )
            for q_idx in range(limit)
        ]

    all_results = run_pool_with_progress(
        build_tasks=_build_train_tasks,
        worker_fn=_train_worker,
        workers=train_workers,
        description="Total Progress",
    )

    # Summary
    success_count = sum(1 for r in all_results.values() if r)
    print("\n=== Training Complete ===")
    print(f"Questions:  {len(all_results)} (Success: {success_count})")


def main():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, default=10, help="Num questions")
    parser.add_argument(
        "--model_config",
        type=str,
        nargs="+",
        choices=ML_MODEL_TYPES,
        help="Model type(s) (multiple allowed, space-separated)",
    )
    parser.add_argument("--seed", type=int, help="Single random seed (mutually exclusive with --seeds)")
    parser.add_argument(
        "--seeds", type=str, help="Comma-separated seed list (e.g., 41,42,43)"
    )
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna")
    parser.add_argument(
        "--experiment", type=str, default="default", help="Experiment ID"
    )
    parser.add_argument("--workers", type=int, help="Workers")
    parser.add_argument(
        "--embed-provider",
        type=str,
        default="openrouter",
        choices=EMBED_PROVIDERS,
        help=f"Embedding provider (choices: {', '.join(EMBED_PROVIDERS)})",
    )
    parser.add_argument(
        "--hard_neg_ratio",
        type=float,
        default=0.6,
        help="Hard negative ratio among negatives (default 0.6)",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="XGBoost device (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--proximal-neg",
        action="store_true",
        default=False,
        help="Enable proximal hard negative mining (40%% proximal + 20%% structural + 40%% random)",
    )
    parser.add_argument(
        "--confidence-weight",
        action="store_true",
        default=False,
        help="Enable annotation confidence weighting (rank/similarity → sample weight)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help="Enable curriculum learning: Phase A (30%% rounds, easy neg) → Phase B (70%% rounds, hard neg)",
    )
    parser.add_argument(
        "--label-suffix",
        type=str,
        default=None,
        help="Label file suffix (e.g., 'augmented' → q{i}_train_labels_augmented.json)",
    )
    parser.add_argument(
        "--nn-device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch neural network device (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip existing model files (per seed and question)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Training data fraction (0.0-1.0), document-level subsampling",
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

    # Parse seeds argument
    seeds_list = None
    if args.seeds:
        seeds_list = [int(s.strip()) for s in args.seeds.split(",")]

    train_models(
        dataset=args.dataset,
        limit=args.limit,
        model_configs=args.model_config,
        experiment=args.experiment,
        use_optuna=not args.no_optuna,
        workers=args.workers,
        provider=args.embed_provider,
        seed=args.seed,
        seeds=seeds_list,
        hard_neg_ratio=args.hard_neg_ratio,
        xgb_device=args.xgb_device,
        proximal_neg=args.proximal_neg,
        confidence_weight=args.confidence_weight,
        curriculum=args.curriculum,
        label_suffix=args.label_suffix,
        nn_device=args.nn_device,
        skip_existing=args.skip_existing,
        train_fraction=args.train_fraction,
        parser=args.struct_parser,
    )


if __name__ == "__main__":
    main()
