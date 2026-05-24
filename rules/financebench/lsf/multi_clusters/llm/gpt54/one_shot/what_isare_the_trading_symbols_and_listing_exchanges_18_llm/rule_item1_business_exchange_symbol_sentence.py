def rule_item1_business_exchange_symbol_sentence(doc: dict) -> list[dict]:
    """Match business-section narrative sentences mentioning exchange and symbol together."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if (
                ("nasdaq" in txt or "stock exchange" in txt)
                and "symbol" in txt
                and span.get("page_no", 999) <= 5
            ):
                out.append(span)
        return out
    except Exception:
        return []
