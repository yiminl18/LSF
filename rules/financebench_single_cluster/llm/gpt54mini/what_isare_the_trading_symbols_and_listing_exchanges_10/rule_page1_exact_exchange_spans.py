def rule_page1_exact_exchange_spans(doc: dict) -> list[dict]:
    """Match page 1 spans whose text is exactly an exchange name."""
    try:
        texts = doc.get("texts", [])
        out = []
        exacts = {
            "new york stock exchange",
            "the new york stock exchange",
            "new york stock exchange (nyse)",
            "nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global select market",
        }
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") == 1 and txt in exacts:
                out.append(span)
        return out
    except Exception:
        return []
