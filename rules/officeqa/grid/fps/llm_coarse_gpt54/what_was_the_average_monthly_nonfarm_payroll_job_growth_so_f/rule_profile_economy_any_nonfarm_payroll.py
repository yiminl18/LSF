def rule_profile_economy_any_nonfarm_payroll(doc: dict) -> list[dict]:
    """Match any Profile of the Economy text span mentioning nonfarm payrolls."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and "Profile of the Economy" in (((span.get("structure") or {}).get("path_text")) or "")
            and re.search(r'nonfarm payroll', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
