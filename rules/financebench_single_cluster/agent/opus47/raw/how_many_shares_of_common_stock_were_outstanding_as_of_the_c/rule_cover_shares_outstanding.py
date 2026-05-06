def rule_cover_shares_outstanding(doc: dict) -> list[dict]:
    """Match page 1-2 spans mentioning shares outstanding or common stock outstanding."""
    try:
        results = []
        for span in doc.get("texts", []):
            page = span.get("page_no", 0)
            if page not in (1, 2):
                continue
            text = span.get("text", "").lower()
            if ("shares" in text and "outstanding" in text) or \
               ("common stock" in text and "outstanding" in text):
                results.append(span)
        return results
    except Exception:
        return []
