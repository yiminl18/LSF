def rule_page1_issued_and_outstanding_sentence(doc: dict) -> list[dict]:
    """Match page-1 spans using the phrase 'issued and outstanding'."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "issued and outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
