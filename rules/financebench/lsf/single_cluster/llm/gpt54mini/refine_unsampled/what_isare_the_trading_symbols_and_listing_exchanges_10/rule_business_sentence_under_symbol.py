def rule_business_sentence_under_symbol(doc: dict) -> list[dict]:
    """Match business-section sentences of the form 'common stock is listed/trades ... under the symbol ...'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if ("under the symbol" in txt or "traded under the symbol" in txt) and (
                "listed on" in txt or "trades on" in txt or "stock exchange" in txt or "nasdaq" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
