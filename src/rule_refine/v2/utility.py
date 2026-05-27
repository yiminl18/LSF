"""Stage C — composite utility u(r).

    u(r) = (proxy_acc^alpha) * (cov^beta) / (cost + eps) * (1 - spec)
"""

from __future__ import annotations


def utility(
    profile_entry: dict,
    *,
    alpha: float = 1.0,
    beta:  float = 1.0,
    eps:   float = 1e-4,
) -> float:
    p = profile_entry
    acc_term  = (p["proxy_acc"] ** alpha) if p["proxy_acc"] > 0 else 0.0
    cov_term  = (p["cov"]       ** beta)  if p["cov"]       > 0 else 0.0
    cost_term = p["cost"] + eps
    spec_term = max(0.0, 1.0 - p.get("spec", 0.0))
    return acc_term * cov_term / cost_term * spec_term


def rank_by_utility(
    profile: dict[str, dict],
    *,
    alpha: float = 1.0,
    beta:  float = 1.0,
    eps:   float = 1e-4,
) -> list[tuple[str, float]]:
    """Return [(rule_name, u(r)), ...] sorted descending by u."""
    ranked = [(r, utility(p, alpha=alpha, beta=beta, eps=eps)) for r, p in profile.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
