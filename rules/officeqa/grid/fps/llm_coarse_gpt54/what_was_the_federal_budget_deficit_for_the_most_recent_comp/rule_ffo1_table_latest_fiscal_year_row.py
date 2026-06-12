def rule_ffo1_table_latest_fiscal_year_row(doc: dict) -> list[dict]:
    """Match FFO-1 tables containing fiscal-year rows and deficit columns."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "summary of fiscal operations" in txt and "fiscal year or month" in txt and "deficit" in txt:
                out.append(span)
    except Exception:
        return []
    return out
