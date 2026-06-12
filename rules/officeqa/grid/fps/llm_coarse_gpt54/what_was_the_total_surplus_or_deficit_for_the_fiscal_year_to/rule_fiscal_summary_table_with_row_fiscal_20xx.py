def rule_fiscal_summary_table_with_row_fiscal_20xx(doc: dict) -> list[dict]:
    """Match tables with a row beginning 'Fiscal 20xx' and deficit columns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r'fiscal\s+20\d{2}', txt, re.I) and re.search(r'(surplus|deficit)', txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
