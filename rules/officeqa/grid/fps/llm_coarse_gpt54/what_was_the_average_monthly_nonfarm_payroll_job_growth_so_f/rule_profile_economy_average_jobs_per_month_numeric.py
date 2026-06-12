def rule_profile_economy_average_jobs_per_month_numeric(doc: dict) -> list[dict]:
    """Match Profile of the Economy text spans containing a numeric 'jobs per month' average."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and "Profile of the Economy" in path
                and re.search(r'\b\d[\d,]*\s+jobs per month\b', txt, re.I)
                and re.search(r'(average|averaged)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
