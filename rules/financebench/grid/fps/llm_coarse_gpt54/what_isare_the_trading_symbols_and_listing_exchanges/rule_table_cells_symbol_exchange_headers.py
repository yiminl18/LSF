def rule_table_cells_symbol_exchange_headers(doc: dict) -> list[dict]:
    """Match tables whose headers include both a symbol column and an exchange column."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table":
                continue
            cells = (((s.get("table_data") or {}).get("cells")) or [])
            headers = [c.get("text") or "" for c in cells if c.get("is_column_header")]
            joined = " | ".join(headers)
            if re.search(r"Trading Symbol|Trading symbol", joined, re.I) and re.search(r"exchange", joined, re.I):
                out.append(s)
        return out
    except Exception:
        return []
