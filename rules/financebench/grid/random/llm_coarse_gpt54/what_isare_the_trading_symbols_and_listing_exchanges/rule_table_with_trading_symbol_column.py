def rule_table_with_trading_symbol_column(doc: dict) -> list[dict]:
    """Match table spans whose cells include a 'Trading Symbol' column header."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(re.search(r"trading symbol", (c.get("text", "") or ""), re.I) for c in cells):
                out.append(span)
        return out
    except Exception:
        return []
