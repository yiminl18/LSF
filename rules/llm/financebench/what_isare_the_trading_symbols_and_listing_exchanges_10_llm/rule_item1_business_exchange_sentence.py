def rule_item1_business_exchange_sentence(doc: dict) -> list[dict]:
    """Match Item 1 / Business spans mentioning exchange names and ticker symbols."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure", {}) or {}).get("path_text") or "").lower()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "item 1" in path and "business" in path:
                if ("nasdaq" in txt or "new york stock exchange" in txt or "nyse" in txt) and ("symbol" in txt or "trades" in txt or "listed" in txt):
                    out.append(span)
        return out
    except Exception:
        return []
