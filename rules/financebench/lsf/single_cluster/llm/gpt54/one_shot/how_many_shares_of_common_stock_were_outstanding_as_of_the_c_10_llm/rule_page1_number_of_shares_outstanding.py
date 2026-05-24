def rule_page1_number_of_shares_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans with the phrase 'number of shares' and 'outstanding'."""
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares" in text and "outstanding" in text:
                out.append(span)
    except Exception:
        return []
    return out
