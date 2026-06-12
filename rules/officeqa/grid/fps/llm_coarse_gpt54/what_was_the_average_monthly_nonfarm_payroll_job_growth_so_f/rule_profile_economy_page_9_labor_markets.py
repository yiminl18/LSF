def rule_profile_economy_page_9_labor_markets(doc: dict) -> list[dict]:
    """Match likely answer spans on page 9 in newer bulletins under Labor Markets and Wages."""
    import re
    try:
        out = []
        current_header = ""
        for span in doc.get("texts", []):
            if span.get("page_no") == 9 and span.get("label") == "section_header":
                current_header = span.get("text") or ""
            txt = span.get("text") or ""
            if (
                span.get("page_no") == 9
                and span.get("label") == "text"
                and re.search(r'Labor Markets and Wages', current_header, re.I)
                and re.search(r'(job growth|payroll job creation|average\s+\d[\d,]*)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
