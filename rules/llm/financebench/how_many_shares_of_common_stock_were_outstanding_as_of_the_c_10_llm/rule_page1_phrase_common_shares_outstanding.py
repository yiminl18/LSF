def rule_page1_phrase_common_shares_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans with 'common shares outstanding' phrasing."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "common shares" in t and "outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
