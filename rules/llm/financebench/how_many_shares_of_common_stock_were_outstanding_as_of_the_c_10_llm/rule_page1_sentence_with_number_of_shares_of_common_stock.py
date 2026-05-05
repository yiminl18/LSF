def rule_page1_sentence_with_number_of_shares_of_common_stock(doc: dict) -> list[dict]:
    """Match page-1 spans using 'Number of shares of common stock outstanding...' wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares of common stock" in t:
                out.append(span)
    except Exception:
        return []
    return out
