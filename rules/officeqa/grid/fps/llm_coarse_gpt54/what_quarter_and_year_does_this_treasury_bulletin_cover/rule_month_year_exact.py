def rule_month_year_exact(doc: dict) -> list[dict]:
    """Match exact month-year spans like 'October 1980' or 'June 2008'."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
            re.I,
        )
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
