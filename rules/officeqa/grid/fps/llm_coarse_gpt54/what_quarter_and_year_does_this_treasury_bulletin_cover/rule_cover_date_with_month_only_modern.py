def rule_cover_date_with_month_only_modern(doc: dict) -> list[dict]:
    """Match modern issue month-year spans even when not adjacent to Treasury Bulletin text."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"\b(March|June|September|December)\s+\d{4}\b", re.I)
        out = []
        for s in texts:
            if s.get("page_no", 999) <= 6 and pat.search((s.get("text") or "").strip()):
                out.append(s)
        return out
    except Exception:
        return []
