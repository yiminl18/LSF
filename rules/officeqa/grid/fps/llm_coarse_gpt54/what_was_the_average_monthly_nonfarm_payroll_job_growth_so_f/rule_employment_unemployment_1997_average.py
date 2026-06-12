def rule_employment_unemployment_1997_average(doc: dict) -> list[dict]:
    """Match the 1997-style Employment and unemployment paragraph with first 10 months average payroll growth."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and "Employment and unemployment" in path
                and re.search(r'first\s+10\s+months', txt, re.I)
                and re.search(r'nonfarm payrolls', txt, re.I)
                and re.search(r'averaged\s+\d[\d,]*\s+per month', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
