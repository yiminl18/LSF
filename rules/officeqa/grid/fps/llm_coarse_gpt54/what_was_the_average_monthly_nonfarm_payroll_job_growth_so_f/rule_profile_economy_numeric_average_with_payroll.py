def rule_profile_economy_numeric_average_with_payroll(doc: dict) -> list[dict]:
    """Match spans with a numeric average and payroll-related wording anywhere in the document."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "text"
            and re.search(r'(average|averaged)\s+[-]?\d[\d,]*', span.get("text") or "", re.I)
            and re.search(r'(nonfarm payroll|payroll job|jobs per month|job growth)', span.get("text") or "", re.I)
        ]
    except Exception:
        return []
