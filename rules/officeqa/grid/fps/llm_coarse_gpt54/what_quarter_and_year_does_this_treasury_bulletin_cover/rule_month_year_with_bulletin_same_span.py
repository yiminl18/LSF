def rule_month_year_with_bulletin_same_span(doc: dict) -> list[dict]:
    """Match spans that combine Treasury Bulletin and the date in one text span."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"treasury\s+bulletin.*\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December"
            r")\s+\d{4}\b",
            re.I | re.S,
        )
        return [s for s in texts if pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
