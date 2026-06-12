def rule_tables_with_row_individual_income_taxes(doc: dict) -> list[dict]:
    """Match tables where a row label explicitly says Individual income taxes."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'Individual income taxes', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
