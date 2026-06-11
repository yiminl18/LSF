def rule_page1_trading_symbol_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Trading Symbol' or 'Trading Symbol(s)'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"trading symbol(?:\(s\))?", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
