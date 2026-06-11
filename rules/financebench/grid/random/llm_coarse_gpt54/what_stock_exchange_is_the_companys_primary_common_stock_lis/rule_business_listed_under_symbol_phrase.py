def rule_business_listed_under_symbol_phrase(doc: dict) -> list[dict]:
    """Match Business-section spans containing 'under the symbol' and an exchange name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if ("item 1" in path or "business" in path) and "under the symbol" in txt and ("nasdaq" in txt or "stock exchange" in txt):
                out.append(span)
        return out
    except Exception:
        return []
