def rule_page1_after_trading_symbol_header(doc: dict) -> list[dict]:
    """Return page-1 spans immediately following a 'Trading Symbol' header span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"trading symbol(?:\(s\))?", txt, re.I):
                for j in range(i + 1, min(i + 6, len(texts))):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
