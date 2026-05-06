def rule_page1_sentence_with_registrant_and_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans containing both 'registrant' and outstanding-share wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "registrant" in t and "outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
