def rule_page1_phrase_common_stock_issued_and_outstanding_as_of(doc: dict) -> list[dict]:
    """Match page-1 spans with 'common stock issued and outstanding as of' phrasing."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "common stock" in t and "issued and outstanding" in t and "as of" in t:
                out.append(span)
    except Exception:
        return []
    return out
