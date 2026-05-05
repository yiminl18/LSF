def rule_page2_issued_and_outstanding_sentence(doc: dict) -> list[dict]:
    """Match page-2 spans using the phrase 'issued and outstanding' for filings where the cover page spills to page 2."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 2 and "issued and outstanding" in t and ("common stock" in t or "shares" in t):
                out.append(span)
    except Exception:
        return []
    return out
