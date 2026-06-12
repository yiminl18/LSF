def rule_profile_economy_average_monthly_job_growth_phrase(doc: dict) -> list[dict]:
    """Match spans explicitly containing the phrase average monthly job growth or close variants."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if (
                span.get("label") == "text"
                and re.search(r'average monthly .*job growth', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
