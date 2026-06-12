def rule_issue_of_quarter_fiscal(doc: dict) -> list[dict]:
    """Match seasonal issue phrases that include quarter and fiscal year."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(Spring|Summer|Fall|Winter)\s+Issue(?:\s+of)?\s+(First|Second|Third|Fourth)\s+Quarter,?\s+Fiscal\s+\d{4}$",
            re.I,
        )
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
