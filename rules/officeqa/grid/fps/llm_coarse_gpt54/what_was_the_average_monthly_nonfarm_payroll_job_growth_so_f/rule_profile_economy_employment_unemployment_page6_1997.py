def rule_profile_economy_employment_unemployment_page6_1997(doc: dict) -> list[dict]:
    """Match 1997-style page 6 employment paragraph with average 239,000 payroll growth."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if (
                span.get("page_no") == 6
                and span.get("label") == "text"
                and re.search(r'nonfarm payrolls averaged', txt, re.I)
                and re.search(r'first 10 months of this year', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
