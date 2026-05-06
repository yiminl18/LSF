def rule_tables_with_total_assets_and_few_columns(doc: dict) -> list[dict]:
    """Match balance-sheet tables with row labels plus a small number of year columns."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            td = span.get("table_data") or {}
            num_cols = td.get("num_cols") or 0
            txt = (span.get("text") or "").lower()
            if "total assets" in txt and 2 <= num_cols <= 6:
                out.append(span)
        return out
    except Exception:
        return []
