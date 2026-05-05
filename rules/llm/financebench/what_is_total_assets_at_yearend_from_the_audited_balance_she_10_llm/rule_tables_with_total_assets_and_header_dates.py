def rule_tables_with_total_assets_and_header_dates(doc: dict) -> list[dict]:
    """Match total-assets tables with date-like column headers."""
    import re
    try:
        out = []
        date_re = re.compile(r"(december|january|february|march|april|may|june|july|august|september|october|november|\b20\d{2}\b|\b19\d{2}\b)", re.I)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            if "total assets" not in (span.get("text") or "").lower():
                continue
            headers = [c for c in (span.get("table_data") or {}).get("cells", []) if c.get("is_column_header")]
            if any(date_re.search(c.get("text") or "") for c in headers):
                out.append(span)
        return out
    except Exception:
        return []
