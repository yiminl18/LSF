def rule_page1_trading_symbol_keyword(doc: dict) -> list[dict]:
    """Match spans on page 1 containing trading symbol keywords."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "trading symbol" in txt or "trading symbol(s)" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
