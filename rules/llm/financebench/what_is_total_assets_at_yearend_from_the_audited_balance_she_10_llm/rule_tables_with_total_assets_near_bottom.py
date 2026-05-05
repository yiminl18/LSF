def rule_tables_with_total_assets_near_bottom(doc: dict) -> list[dict]:
    """Match tables where total assets appears in later rows, as is typical in balance sheets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            td = span.get("table_data") or {}
            num_rows = td.get("num_rows") or 0
            cells = td.get("cells") or []
            for c in cells:
                txt = (c.get("text") or "").lower()
                if "total assets" in txt and num_rows and c.get("row", 0) >= max(1, int(num_rows * 0.3)):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
