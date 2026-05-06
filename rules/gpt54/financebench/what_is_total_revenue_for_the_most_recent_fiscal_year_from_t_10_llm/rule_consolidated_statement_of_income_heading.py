def rule_consolidated_statement_of_income_heading(doc: dict) -> list[dict]:
    """Match section headers naming Consolidated Statement(s) of Income."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if "consolidated statement of income" in txt or "consolidated statements of income" in txt:
                out.append(span)
        return out
    except Exception:
        return []
