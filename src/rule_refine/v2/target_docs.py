"""Stage B — define the target document set D*.

D* is the proxy ceiling: docs where at least one rule retrieves text containing
the ground-truth answer string. The selection algorithm tries to cover D*.
"""

from __future__ import annotations


def build_target_set(profile: dict[str, dict]) -> set[str]:
    """Return D* = ⋃_r proxy_docs(r)."""
    target: set[str] = set()
    for prof in profile.values():
        target |= prof["proxy_docs"]
    return target


def filter_useful_rules(profile: dict[str, dict], target_docs: set[str]) -> list[str]:
    """Return rules whose proxy_docs intersect D* (the rest can never help)."""
    return [r for r, p in profile.items() if p["proxy_docs"] & target_docs]
