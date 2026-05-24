def rule_item1_business_stock_listed(doc: dict) -> list[dict]:
    """Match Item 1/Business spans that state stock trades/listed under a symbol."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if (
                "item 1" in path or "business" in path
            ) and (
                "listed on" in txt and "symbol" in txt
                or "trades on" in txt and "symbol" in txt
                or "common stock is listed on" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
