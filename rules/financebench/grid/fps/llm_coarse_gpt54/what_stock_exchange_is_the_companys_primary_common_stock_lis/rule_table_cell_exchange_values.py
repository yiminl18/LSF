def rule_table_cell_exchange_values(doc: dict) -> list[dict]:
    """Return synthetic matches as the parent table span when table cells contain exchange values."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in (((span.get("table_data") or {}).get("cells")) or []):
                txt = (c.get("text") or "").strip()
                if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ$', txt, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
