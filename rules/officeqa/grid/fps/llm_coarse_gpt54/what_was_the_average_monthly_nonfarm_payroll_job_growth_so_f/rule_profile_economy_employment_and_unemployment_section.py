def rule_profile_economy_employment_and_unemployment_section(doc: dict) -> list[dict]:
    """Match spans under Employment and unemployment / Labor Markets and Wages that mention average payroll job growth."""
    import re
    try:
        out = []
        current_header = ""
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                current_header = span.get("text") or ""
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "text":
                if (
                    ("Profile of the Economy" in path or re.search(r'profile of the economy', path, re.I))
                    and re.search(r'(Employment and unemployment|Labor Markets and Wages|Labor Markets)', current_header, re.I)
                    and re.search(r'(nonfarm payroll|payroll job)', txt, re.I)
                    and re.search(r'(average|averaged)', txt, re.I)
                    and re.search(r'(this year|first \d+ months|thus far in 2024|current calendar year)', txt, re.I)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
