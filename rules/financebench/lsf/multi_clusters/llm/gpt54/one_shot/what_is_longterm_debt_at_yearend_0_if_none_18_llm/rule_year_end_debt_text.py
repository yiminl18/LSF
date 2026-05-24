def rule_year_end_debt_text(doc: dict) -> list[dict]:
    """Match spans mentioning debt at year-end or period-end."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r"(year[- ]end|period[- ]end|at\s+.*end).*debt|debt.*(year[- ]end|period[- ]end)", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
