def rule_page1_after_market_value_sentence(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning outstanding shares that appear after a market-value span."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_market = False
        for span in texts:
            if span.get("page_no") not in (1, 2):
                continue
            t = (span.get("text") or "").lower()
            if "aggregate market value" in t or "market value of" in t:
                seen_market = True
            if seen_market and "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
        return out
    except Exception:
        return []
