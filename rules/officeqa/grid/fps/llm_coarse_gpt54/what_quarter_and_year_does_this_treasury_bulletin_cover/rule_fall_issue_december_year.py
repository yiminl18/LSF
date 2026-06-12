def rule_fall_issue_december_year(doc: dict) -> list[dict]:
    """Match later pattern 'Fall Issue December YYYY' or 'FALL ISSUE, DECEMBER YYYY'."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"^Fall Issue December \d{4}$|^FALL ISSUE,?\s+DECEMBER\s+\d{4}$", re.I)
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
