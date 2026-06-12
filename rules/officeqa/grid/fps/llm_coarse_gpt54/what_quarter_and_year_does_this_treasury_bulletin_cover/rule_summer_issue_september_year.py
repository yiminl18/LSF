def rule_summer_issue_september_year(doc: dict) -> list[dict]:
    """Match combined summer issue + September year strings."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"SUMMER\s+ISSUE,?\s+SEPTEMBER\s+\d{4}", re.I)
        return [s for s in texts if pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
