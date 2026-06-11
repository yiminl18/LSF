def rule_exchange_in_table_with_trading_symbol_header(doc: dict) -> list[dict]:
    """Match table spans whose headers include trading symbol and exchange registration columns."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " | ".join((c.get("text") or "") for c in cells)
            if re.search(r'trading symbol', joined, re.I) and re.search(r'exchange', joined, re.I):
                if re.search(r'new york stock exchange|nasdaq|global select market', joined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
