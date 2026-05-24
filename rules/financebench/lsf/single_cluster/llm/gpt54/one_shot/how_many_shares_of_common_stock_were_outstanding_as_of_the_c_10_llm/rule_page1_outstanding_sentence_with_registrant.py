def rule_page1_outstanding_sentence_with_registrant(doc: dict) -> list[dict]:
    """Match page-1/2 spans mentioning both 'registrant' and 'outstanding'."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") in (1, 2) and "registrant" in t and "outstanding" in t:
                out.append(span)
        return out
    except Exception:
        return []
