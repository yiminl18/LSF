def rule_page1_cover_page_stock_sentence(doc: dict) -> list[dict]:
    """Match likely cover-page stock-count sentences on page 1 with 'as of' and 'outstanding'."""
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            t = text.lower()
            if span.get("page_no") == 1 and "as of" in t and "outstanding" in t:
                if "common stock" in t or "common shares" in t or "shares of common stock" in t:
                    out.append(span)
    except Exception:
        return []
    return out
