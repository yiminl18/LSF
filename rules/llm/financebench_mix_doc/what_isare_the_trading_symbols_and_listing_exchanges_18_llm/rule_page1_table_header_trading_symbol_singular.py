def rule_page1_table_header_trading_symbol_singular(doc: dict) -> list[dict]:
    """Match spans containing the exact singular header 'Trading Symbol'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "trading symbol" in txt:
                out.append(span)
        return out
    except Exception:
        return []
