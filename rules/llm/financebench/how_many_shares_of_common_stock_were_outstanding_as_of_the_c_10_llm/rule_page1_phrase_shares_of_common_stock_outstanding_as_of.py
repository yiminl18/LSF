def rule_page1_phrase_shares_of_common_stock_outstanding_as_of(doc: dict) -> list[dict]:
    """Match page-1 spans with 'shares of common stock outstanding as of' phrasing."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "shares of common stock" in t and "outstanding as of" in t:
                out.append(span)
    except Exception:
        return []
    return out
