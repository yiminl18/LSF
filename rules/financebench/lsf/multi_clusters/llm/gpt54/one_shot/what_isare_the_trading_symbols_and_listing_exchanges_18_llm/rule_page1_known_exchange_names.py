def rule_page1_known_exchange_names(doc: dict) -> list[dict]:
    """Match page-1 spans that look like exchange names."""
    try:
        texts = doc.get("texts", [])
        out = []
        exchange_terms = [
            "new york stock exchange",
            "nasdaq",
            "nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global market",
            "the nasdaq global select",
            "chicago stock exchange",
            "australian securities exchange",
        ]
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and any(term in txt for term in exchange_terms):
                out.append(span)
        return out
    except Exception:
        return []
