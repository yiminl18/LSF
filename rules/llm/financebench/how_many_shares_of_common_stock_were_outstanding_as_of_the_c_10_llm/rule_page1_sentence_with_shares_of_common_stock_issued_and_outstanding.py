def rule_page1_sentence_with_shares_of_common_stock_issued_and_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans using 'shares of common stock issued and outstanding...' wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "shares of common stock" in t and "issued and outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
