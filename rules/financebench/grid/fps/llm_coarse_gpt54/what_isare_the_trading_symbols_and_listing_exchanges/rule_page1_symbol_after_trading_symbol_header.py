def rule_page1_symbol_after_trading_symbol_header(doc: dict) -> list[dict]:
    """Match spans immediately following a 'Trading Symbol' header on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"Trading\s+Symbol(?:\(s\))?", (s.get("text") or ""), re.I):
                for j in range(i + 1, min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
