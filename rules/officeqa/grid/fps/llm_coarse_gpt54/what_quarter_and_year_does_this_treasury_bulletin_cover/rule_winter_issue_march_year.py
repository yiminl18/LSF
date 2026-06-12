def rule_winter_issue_march_year(doc: dict) -> list[dict]:
    """Match combined winter issue + month/year title strings."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"WINTER\s+ISSUE.*MARCH\s+\d{4}", re.I)
        return [s for s in texts if pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
