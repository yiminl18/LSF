def rule_profile_economy_labor_markets_page9_2024(doc: dict) -> list[dict]:
    """Match 2024-style page 9 labor markets paragraph with average 203,000 jobs."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if (
                span.get("page_no") == 9
                and span.get("label") == "text"
                and re.search(r'In 2024 thus far', txt, re.I)
                and re.search(r'job growth has average[d]?\s+\d[\d,]*', txt, re.I)
                and re.search(r'adequate', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
