# -*- coding: utf-8 -*-
"""
Hybrid Neural Network (HNN) classifier.

A PyTorch-based shallow MLP classifier with Focal Loss training.
Designed for low-data regimes (500-2000 samples per query).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.ml.base import BaseClassifier
from core.ml.metrics import compute_mrr_for_eval

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive samples
    FL(p) = -(1-alpha) * p^gamma * log(1-p)  for negative samples
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model logits (pre-sigmoid)
            targets: Ground-truth labels (0/1)
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        probs = torch.sigmoid(inputs)
        # Positive: alpha * (1-p)^gamma, Negative: (1-alpha) * p^gamma
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class ResidualBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Projection for dimension mismatch
        self.proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.dropout(self.act(self.norm(self.linear(x))))
        return out + residual


class ShallowMLP(nn.Module):
    """MLP network with optional residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        use_residual: bool = False,
        use_input_bn: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.input_norm = nn.BatchNorm1d(input_dim) if use_input_bn else nn.Identity()
        self.feature_attention: Optional[nn.MultiheadAttention] = None

        if use_residual:
            blocks = []
            prev_dim = input_dim
            for i, hidden_dim in enumerate(hidden_dims):
                drop_rate = dropout * (1 - i * 0.3)
                blocks.append(ResidualBlock(prev_dim, hidden_dim, max(drop_rate, 0)))
                prev_dim = hidden_dim
            self.blocks = nn.ModuleList(blocks)
            if hidden_dims:
                self.feature_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dims[0],
                    num_heads=1,
                    batch_first=True,
                )
            self.output = nn.Linear(prev_dim, 1)
        else:
            blocks = []
            prev_dim = input_dim
            for i, hidden_dim in enumerate(hidden_dims):
                layers = [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
                drop_rate = dropout * (1 - i * 0.3)
                if drop_rate > 0:
                    layers.append(nn.Dropout(drop_rate))
                blocks.append(nn.Sequential(*layers))
                prev_dim = hidden_dim
            self.blocks = nn.ModuleList(blocks)
            if hidden_dims:
                self.feature_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dims[0],
                    num_heads=1,
                    batch_first=True,
                )
            self.output = nn.Linear(prev_dim, 1)

    def _apply_feature_attention(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_attention is None:
            return x
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.feature_attention(x_seq, x_seq, x_seq, need_weights=False)
        return x + attn_out.squeeze(1)

    def _safe_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Safe BatchNorm: temporarily switch to eval for single-sample training (uses running stats)."""
        if (
            x.size(0) == 1
            and self.training
            and isinstance(self.input_norm, nn.BatchNorm1d)
        ):
            self.input_norm.eval()
            x = self.input_norm(x)
            self.input_norm.train()
            return x
        return self.input_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._safe_bn(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0:
                x = self._apply_feature_attention(x)
        return self.output(x).squeeze(-1)


class HybridNNClassifier(BaseClassifier):
    """
    Hybrid Neural Network classifier.

    Features:
    - Shallow MLP (2 hidden layers) to prevent overfitting on small data
    - Focal Loss for class imbalance
    - MRR early stopping
    - Group-aware train/val split to preserve data integrity
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required: pip install torch")
        super().__init__(params, feature_names)
        self.model: Optional[ShallowMLP] = None
        self.input_dim: Optional[int] = None
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

    @classmethod
    def get_classifier_type(cls) -> str:
        return "hnn"

    def get_default_params(self) -> Dict:
        return {
            "hidden_dims": (128, 64, 32),
            "dropout": 0.3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "loss_type": "focal",  # 'focal', 'bce'
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "infonce_tau": 1.0,
            "infonce_accum": 16,  # Listwise gradient accumulation steps
            "approxndcg_temperature": 0.1,
            "listmle_temperature": 1.0,
            "use_residual": True,
            "seed": 42,
        }

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        doc_ids: Optional[List[str]] = None,
        group_keys: Optional[List[str]] = None,
        num_rounds: int = 100,
        early_stopping_rounds: Optional[int] = 10,
        val_split: float = 0.2,
        verbose: bool = True,
        use_mrr: bool = True,
        sample_weight: Optional[np.ndarray] = None,
        init_state_dict: Optional[Dict] = None,
    ) -> Dict[str, float]:
        # Set random seed
        seed = self.params.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.input_dim = X.shape[1]
        # Build group info
        if group_keys is None:
            group_keys = [d.rsplit("_", 1)[0] if d else "" for d in (doc_ids or [])]
        group_indices = self._build_group_indices(group_keys)
        # Group-aware train/val split
        train_idx, val_idx = self._group_split(group_indices, val_split, seed)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_doc_ids = [doc_ids[i] for i in val_idx] if doc_ids else None
        # Convert to Tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        # Create DataLoader
        configured_batch_size = max(1, int(self.params.get("batch_size", 128)))
        batch_size = min(configured_batch_size, max(1, len(X_train) // 4))
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Initialize model
        hidden_dims = self.params.get("hidden_dims", (128, 64, 32))
        dropout = self.params.get("dropout", 0.3)
        use_residual = self.params.get("use_residual", True)
        self.model = ShallowMLP(self.input_dim, hidden_dims, dropout, use_residual).to(
            self.device
        )

        # Curriculum learning warm-start: continue training from previous phase weights
        if init_state_dict is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in init_state_dict.items()}
            )
        assert self.model is not None

        def _forward(x: torch.Tensor) -> torch.Tensor:
            assert self.model is not None
            return self.model(x)

        # Loss function
        loss_type = self.params.get("loss_type", "focal")
        # Optimizer
        lr = self.params.get("learning_rate", 1e-3)
        weight_decay = self.params.get("weight_decay", 1e-4)
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

        # Early stopping
        patience = early_stopping_rounds or 10
        best_metric = -float("inf")
        best_epoch = 0
        best_state = None

        # -- Pointwise training (focal / bce) --
        if loss_type == "focal":
            alpha = self.params.get("focal_alpha", 0.25)
            gamma = self.params.get("focal_gamma", 2.0)
            criterion = FocalLoss(alpha=alpha, gamma=gamma)
        else:
            criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_rounds):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                y_smooth = batch_y * 0.9 + 0.05
                outputs = _forward(batch_X)
                loss = criterion(outputs, y_smooth)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = _forward(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_probs = torch.sigmoid(val_outputs).cpu().numpy()

            if use_mrr and val_doc_ids:
                val_mrr = compute_mrr_for_eval(y_val, val_probs, val_doc_ids)
                current_metric = val_mrr
                metric_name = "MRR"
            else:
                current_metric = -val_loss
                metric_name = "Loss"

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            if verbose and epoch % 10 == 0:
                if use_mrr and val_doc_ids:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mrr={val_mrr:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )

            if epoch - best_epoch >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}, best {metric_name} at epoch {best_epoch}"
                    )
                break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        # Final metrics
        self.model.eval()
        with torch.no_grad():
            val_outputs = _forward(X_val_t)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()

        metrics = {
            "best_iteration": best_epoch,
        }

        if use_mrr and val_doc_ids:
            final_mrr = compute_mrr_for_eval(y_val, val_probs, val_doc_ids)
            metrics["val_mrr"] = final_mrr
        return metrics

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Batch prediction, returns [0,1] probabilities."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.sigmoid(outputs).cpu().numpy()

        return probs

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (based on absolute input-layer weights).

        Note: MLP feature importance is less intuitive than tree models;
        use only as a rough reference.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        if self.feature_names is None:
            raise ValueError("Feature names not set")

        # Use mean absolute first-layer weights as importance
        if self.model.use_residual:
            first_layer = self.model.blocks[0].linear
        else:
            first_layer = self.model.blocks[0][0]
        weights = first_layer.weight.data.cpu().numpy()
        importance = np.abs(weights).mean(axis=0)

        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}

    def _sample_from_bn(self, n_samples: int = 200) -> np.ndarray:
        """Sample representative inputs from BatchNorm running statistics."""
        assert self.model is not None
        bn = self.model.input_norm

        if isinstance(bn, nn.BatchNorm1d):
            mean = bn.running_mean.cpu().numpy()
            std = np.sqrt(bn.running_var.cpu().numpy() + bn.eps)
            rng = np.random.RandomState(42)
            return rng.normal(mean, std, (n_samples, len(mean))).astype(np.float32)

        return (
            np.random.RandomState(42)
            .randn(n_samples, self.input_dim)
            .astype(np.float32)
        )

    def get_feature_importance_ig(
        self,
        X: Optional[np.ndarray] = None,
        n_steps: int = 300,
        n_samples: int = 200,
    ) -> Dict[str, float]:
        """Integrated Gradients feature importance.

        Integrates gradients along the zero-baseline-to-input interpolation path,
        quantifying each feature's marginal contribution to the output.
        More accurate than |W|: accounts for non-linearity, residual connections,
        LayerNorm effects.

        Args:
            X: Input samples (n, d). If None, auto-sampled from BatchNorm statistics.
            n_steps: Interpolation steps (Riemann approximation precision)
            n_samples: Auto-sample count (only when X=None)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        if self.feature_names is None:
            raise ValueError("Feature names not set")

        self.model.eval()

        if X is None:
            X = self._sample_from_bn(n_samples)

        X_t = torch.FloatTensor(X).to(self.device)

        # BN-mean baseline = zero baseline in post-BN normalized space (Sundararajan et al., 2017)
        # BN(mu) = (mu-mu)/sigma = 0, satisfying IG's "no-signal" axiom
        bn = self.model.input_norm
        if isinstance(bn, nn.BatchNorm1d):
            bn_mean = bn.running_mean.cpu().numpy().astype(np.float32)
            baseline = (
                torch.FloatTensor(bn_mean).unsqueeze(0).expand_as(X_t).to(self.device)
            )
        else:
            baseline = torch.zeros_like(X_t)

        # Riemann right-endpoint approx: IG = (x - x') * (1/m) * sum dF/dx(x' + k/m * (x - x'))
        ig_sum = torch.zeros_like(X_t)
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interp = (baseline + alpha * (X_t - baseline)).detach().requires_grad_(True)
            self.model.zero_grad()
            out = self.model(interp)

            out.sum().backward()
            ig_sum += interp.grad.detach()

        ig = (X_t - baseline).detach() * ig_sum / n_steps

        # Completeness check: sum(IG_i) ~ F(x) - F(baseline)
        # Use aggregated MAE / MAD to avoid small-denominator amplification
        with torch.no_grad():
            f_x = self.model(X_t)
            f_b = self.model(baseline)
            expected_diff = (f_x - f_b).squeeze(-1)
            actual_sum = ig.sum(dim=1)
            mae = (expected_diff - actual_sum).abs().mean().item()
            mad = expected_diff.abs().mean().item()
            rel_error = mae / max(mad, 1e-8)
            if rel_error > 0.10:
                import warnings

                warnings.warn(
                    f"IG completeness: MAE/MAD = {rel_error:.4f} (>{0.10}). "
                    f"Consider increasing n_steps (current: {n_steps})."
                )

        importance = ig.abs().mean(dim=0).cpu().numpy()

        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), str(save_path))

        # Save metadata
        meta_path = save_path.with_suffix(".meta.json")
        meta = {
            "classifier_type": self.get_classifier_type(),
            "params": self.params,
            "feature_names": self.feature_names,
            "input_dim": self.input_dim,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load model."""
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # Load metadata
        meta_path = load_path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.params = meta.get("params", self.params)
            self.feature_names = meta.get("feature_names")
            self.input_dim = meta.get("input_dim")

        if self.input_dim is None:
            raise ValueError("Cannot determine input dimension; metadata file may be corrupted")

        # Rebuild network architecture
        hidden_dims = self.params.get("hidden_dims", (128, 64, 32))
        dropout = self.params.get("dropout", 0.3)
        use_residual = self.params.get("use_residual", True)

        self.model = ShallowMLP(self.input_dim, hidden_dims, dropout, use_residual).to(
            self.device
        )

        # Load weights
        state_dict = torch.load(str(load_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # _build_group_indices and _group_split inherited from BaseClassifier
