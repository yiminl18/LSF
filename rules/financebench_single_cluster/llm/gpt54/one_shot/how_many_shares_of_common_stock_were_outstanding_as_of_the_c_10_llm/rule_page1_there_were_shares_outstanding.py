def rule_page1_there_were_shares_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans beginning with 'There were ... shares ... outstanding'."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "there were" in t and "outstanding" in t and "shares" in t:
                out.append(span)
    except Exception:
        return []
    return out
