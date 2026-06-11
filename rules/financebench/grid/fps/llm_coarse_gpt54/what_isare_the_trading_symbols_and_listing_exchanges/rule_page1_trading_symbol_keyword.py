def rule_page1_trading_symbol_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Trading Symbol' or 'Trading symbol(s)'."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and re.search(r"Trading\s+Symbol(?:\(s\))?", (s.get("text") or ""), re.I)
        ]
    except Exception:
        return []
