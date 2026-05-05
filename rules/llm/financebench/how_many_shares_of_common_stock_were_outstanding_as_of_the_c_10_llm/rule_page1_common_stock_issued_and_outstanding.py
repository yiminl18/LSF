def rule_page1_common_stock_issued_and_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning 'shares of common stock issued and outstanding'."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "shares of common stock" in t and "issued and outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
