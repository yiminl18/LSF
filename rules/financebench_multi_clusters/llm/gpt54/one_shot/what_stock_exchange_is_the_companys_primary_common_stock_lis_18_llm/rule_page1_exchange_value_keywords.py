def rule_page1_exchange_value_keywords(doc: dict) -> list[dict]:
    """Match page-1 spans whose text looks like a stock exchange name."""
    try:
        out = []
        keys = [
            "new york stock exchange",
            "nasdaq",
            "nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global market",
            "the nasdaq global select",
            "the nasdaq global",
            "the new york stock exchange",
        ]
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower().strip()
            if span.get("page_no") == 1 and any(k in txt for k in keys):
                out.append(span)
        return out
    except Exception:
        return []
