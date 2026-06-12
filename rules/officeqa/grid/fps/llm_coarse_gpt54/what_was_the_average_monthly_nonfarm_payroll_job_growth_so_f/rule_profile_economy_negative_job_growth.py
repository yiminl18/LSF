def rule_profile_economy_negative_job_growth(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans with negative average payroll growth values."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and "Profile of the Economy" in path
                and re.search(r'(-\d[\d,]*|\bminus\s+\d[\d,]*)', txt, re.I)
                and re.search(r'(payroll|job growth|jobs per month)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
