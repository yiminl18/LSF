def rule_page1_registrant_common_stock_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning registrant's common stock outstanding."""
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "registrant" in text and "common stock" in text and "outstanding" in text:
                out.append(span)
    except Exception:
        return []
    return out
