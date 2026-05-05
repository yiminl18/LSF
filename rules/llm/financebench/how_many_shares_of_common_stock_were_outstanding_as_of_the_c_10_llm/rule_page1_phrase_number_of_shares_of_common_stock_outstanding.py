def rule_page1_phrase_number_of_shares_of_common_stock_outstanding(doc: dict) -> list[dict]:
    """Match exact-style phrase 'number of shares of common stock outstanding' on page 1."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares of common stock outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
