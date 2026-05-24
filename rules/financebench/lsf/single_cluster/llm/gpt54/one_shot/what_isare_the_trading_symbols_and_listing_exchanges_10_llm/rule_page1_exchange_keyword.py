def rule_page1_exchange_keyword(doc: dict) -> list[dict]:
    """Match page 1 spans mentioning exchange registration keywords."""
    try:
        texts = doc.get("texts", [])
        out = []
        kws = [
            "name of each exchange",
            "name of each exchange on which registered",
            "exchange on which registered",
            "listing exchange",
            "nasdaq",
            "new york stock exchange",
            "nyse",
            "trading symbol",
            "trading symbol(s)",
        ]
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and any(k in txt for k in kws):
                out.append(span)
        return out
    except Exception:
        return []
