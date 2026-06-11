def rule_exchange_in_registered_pursuant_12b_table_row(doc: dict) -> list[dict]:
    """Match table rows under Section 12(b) registration that contain the exchange value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c.get("text") or "")
            for row_texts in rows.values():
                row = " | ".join(row_texts)
                if re.search(r'common stock|class a common stock', row, re.I) and re.search(r'new york stock exchange|nasdaq|global select market', row, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
