def rule_page1_exchange_value_short_text(doc: dict) -> list[dict]:
    """Match short page-1 text spans that exactly equal common exchange answers."""
    try:
        texts = doc.get("texts", [])
        vals = {
            "new york stock exchange",
            "the new york stock exchange",
            "nasdaq",
            "nasdaq global select market",
            "the nasdaq global select market",
            "the nasdaq global market",
            "nasdaq global market",
        }
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") == 1 and txt in vals:
                out.append(span)
        return out
    except Exception:
        return []
