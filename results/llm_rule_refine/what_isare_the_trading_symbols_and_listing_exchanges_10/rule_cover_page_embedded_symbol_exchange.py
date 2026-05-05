def rule_cover_page_embedded_symbol_exchange(doc: dict) -> list[dict]:
    """Match page 1 spans whose text or text_span embeds both a symbol cue and an exchange cue."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            has_symbol = "trading symbol" in txt or "under the symbol" in txt or "symbol" in txt
            has_exchange = "exchange" in txt or "nasdaq" in txt or "new york stock exchange" in txt or "nyse" in txt
            if has_symbol and has_exchange:
                out.append(span)
        return out
    except Exception:
        return []
