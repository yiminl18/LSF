def rule_total_assets_row_header_first_col(doc: dict) -> list[dict]:
    """Match tables where first-column/row-header cells contain total assets."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                txt = (c.get("text") or "").strip().lower()
                if "total assets" in txt and (c.get("col") == 0 or c.get("is_row_header")):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
