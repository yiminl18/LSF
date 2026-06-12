def rule_profile_economy_current_year_labor_average(doc: dict) -> list[dict]:
    """Match labor-related spans that mention the current year and an average numeric value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and "Profile of the Economy" in path
                and re.search(r'(this year|current year|thus far in \d{4}|first \d+ months of (?:this|the) year)', txt, re.I)
                and re.search(r'(average|averaged)\s+[-]?\d[\d,]*', txt, re.I)
                and re.search(r'(job|payroll|employment)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
