"""Stage D — greedy set-cover with cost-effectiveness.

Iteratively pick the rule that adds the most uncovered D* docs per unit cost,
breaking ties by utility. Stops when D* is fully covered or no rule adds new
docs.
"""

from __future__ import annotations

from .utility import utility


def greedy_cover(
    profile: dict[str, dict],
    target_docs: set[str],
    *,
    alpha: float = 1.0,
    beta:  float = 1.0,
    eps:   float = 1e-4,
) -> tuple[list[str], list[dict]]:
    """Return (selected_rules, trace).

    Each trace entry: {step, picked, gain, uncovered_before, cost, utility}.
    """
    selected: list[str] = []
    uncovered = set(target_docs)
    pool = set(profile.keys())
    trace: list[dict] = []
    step = 0

    while uncovered and pool:
        step += 1
        best_rule = None
        best_score = -1.0
        best_gain = 0
        best_cost = 0.0

        for r in pool:
            p = profile[r]
            gain = len(p["proxy_docs"] & uncovered)
            if gain == 0:
                continue
            score = gain / (p["cost"] + eps)
            # Tie-break by utility
            if score > best_score or (
                score == best_score
                and best_rule is not None
                and utility(profile[r], alpha=alpha, beta=beta, eps=eps)
                  > utility(profile[best_rule], alpha=alpha, beta=beta, eps=eps)
            ):
                best_rule  = r
                best_score = score
                best_gain  = gain
                best_cost  = p["cost"]

        if best_rule is None:
            break

        selected.append(best_rule)
        pool.remove(best_rule)
        newly_covered = profile[best_rule]["proxy_docs"] & uncovered
        uncovered -= newly_covered
        trace.append({
            "step":              step,
            "picked":            best_rule,
            "gain":              best_gain,
            "newly_covered":     sorted(newly_covered),
            "uncovered_after":   sorted(uncovered),
            "cost":              round(best_cost, 6),
            "utility":           round(utility(profile[best_rule], alpha=alpha, beta=beta, eps=eps), 4),
        })

    return selected, trace
