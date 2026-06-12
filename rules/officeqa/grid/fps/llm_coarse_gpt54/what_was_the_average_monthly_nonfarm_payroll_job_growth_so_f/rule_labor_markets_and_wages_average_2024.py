def rule_labor_markets_and_wages_average_2024(doc: dict) -> list[dict]:
    """Match the 2024-style Labor Markets and Wages paragraph with average payroll growth thus far in the year."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("label") == "text"
                and "Labor Markets and Wages" in path
                and re.search(r'In 2024 thus far', txt, re.I)
                and re.search(r'job growth has average[d]?\s+\d[\d,]*', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
