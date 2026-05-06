def rule_tables_with_total_assets_and_dates_in_headers(doc: dict) -> list[dict]:
    """Match tables where total assets appears and header cells look like dates/years."""
    import re
    try:
        out = []
        date_re = re.compile(r"(december|january|february|march|april|may|june|july|august|september|october|november|\b20\d{2}\b|\b19\d{2}\b)", re.I)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (span.get("table_data") or {}).get("cells") or []
            if not any("total assets" in (c.get("text") or "").lower() for c in cells):
                continue
            header_hits = 0
            for c in cells:
                if c.get("is_column_header") and date_re.search(c.get("text") or ""):
                    header_hits += 1
            if header_hits >= 1:
                out.append(span)
        return out
    except Exception:
        return []
