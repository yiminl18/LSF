def rule_recent_pages_with_net_income_mentions(doc: dict) -> list[dict]:
    """Match spans mentioning net income/earnings/loss on later pages where financial statements usually appear."""
    import re
    try:
        max_page = max((s.get("page_no", 0) for s in doc.get("texts", [])), default=0)
        threshold = max(1, int(max_page * 0.2))
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no", 0) >= threshold
            and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
