def rule_page1_table_header_trading_symbols_plural(doc: dict) -> list[dict]:
    """Match spans containing the exact plural header 'Trading Symbol(s)'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") == 1 and "trading symbol(s)" in (span.get("text") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
