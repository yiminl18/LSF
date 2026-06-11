def rule_table_cells_common_stock_rows(doc: dict) -> list[dict]:
    """Match tables containing rows for common stock with nearby symbol/exchange columns."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            cells = (((s.get("table_data") or {}).get("cells")) or [])
            joined = " | ".join((c.get("text") or "") for c in cells)
            if re.search(r"Common Stock", joined, re.I) and re.search(r"(NASDAQ|NYSE|New York Stock Exchange|Nasdaq)", joined, re.I):
                out.append(s)
        return out
    except Exception:
        return []
