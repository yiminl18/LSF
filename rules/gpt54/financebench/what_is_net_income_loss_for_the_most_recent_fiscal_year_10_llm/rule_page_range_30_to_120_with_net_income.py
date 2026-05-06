def rule_page_range_30_to_120_with_net_income(doc: dict) -> list[dict]:
    """Match net income mentions on mid/late pages where 10-K financial sections commonly appear."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if 30 <= (s.get("page_no") or 0) <= 120
            and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
