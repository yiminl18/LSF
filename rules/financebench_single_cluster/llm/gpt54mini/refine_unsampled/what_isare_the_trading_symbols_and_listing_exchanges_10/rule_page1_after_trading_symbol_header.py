def rule_page1_after_trading_symbol_header(doc: dict) -> list[dict]:
    """Match spans immediately following a 'Trading Symbol' header on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "trading symbol" in txt:
                for j in range(i + 1, min(len(texts), i + 5)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
