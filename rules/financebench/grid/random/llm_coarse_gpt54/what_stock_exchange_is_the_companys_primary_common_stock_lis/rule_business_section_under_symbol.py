def rule_business_section_under_symbol(doc: dict) -> list[dict]:
    """Match Business-section spans mentioning both a trading symbol and an exchange."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if ("item 1" in path or "business" in path) and "symbol" in txt and (
                "nasdaq" in txt or "stock exchange" in txt or "nyse" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
