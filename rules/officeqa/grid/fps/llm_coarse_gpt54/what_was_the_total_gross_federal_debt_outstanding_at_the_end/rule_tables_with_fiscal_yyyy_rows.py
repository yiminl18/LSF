def rule_tables_with_fiscal_yyyy_rows(doc: dict) -> list[dict]:
    """Match tables containing rows like 'Fiscal 1984' or 'Fiscal 1982 to date'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r'Fiscal\s+19\d{2}|Fiscal\s+20\d{2}|Fiscal\s+\d{4}\s+to\s+date', text, re.I):
                out.append(span)
    except Exception:
        return []
    return out
