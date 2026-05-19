def rule_page1_market_value_and_number_of_shares_labels(doc: dict) -> list[dict]:
    """Match page-1 spans that are labels for market value and number of shares in side-by-side cover layouts."""
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares of common stock outstanding" in text:
                out.append(span)
    except Exception:
        return []
    return out
