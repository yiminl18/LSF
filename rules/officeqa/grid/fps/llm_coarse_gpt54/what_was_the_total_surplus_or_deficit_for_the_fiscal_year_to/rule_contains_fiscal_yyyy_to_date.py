def rule_contains_fiscal_yyyy_to_date(doc: dict) -> list[dict]:
    """Match tables with a row like 'Fiscal 1982 to date' or similar."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'fiscal\s+\d{4}\s+to\s+date', txt, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
