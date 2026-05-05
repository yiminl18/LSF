def rule_page1_shares_outstanding_as_of(doc: dict) -> list[dict]:
    """Match page-1 spans stating shares were outstanding 'as of' a date."""
    out = []
    try:
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "shares" in text and "outstanding as of" in text:
                out.append(span)
    except Exception:
        return []
    return out
