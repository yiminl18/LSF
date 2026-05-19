def rule_page1_major_exchange_names(doc: dict) -> list[dict]:
    """Match page 1 spans containing common exchange names."""
    try:
        texts = doc.get("texts", [])
        out = []
        exchanges = [
            "new york stock exchange",
            "nasdaq global select market",
            "the nasdaq global select market",
            "the new york stock exchange",
            "new york stock exchange (nyse)",
        ]
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and any(x in txt for x in exchanges):
                out.append(span)
        return out
    except Exception:
        return []
