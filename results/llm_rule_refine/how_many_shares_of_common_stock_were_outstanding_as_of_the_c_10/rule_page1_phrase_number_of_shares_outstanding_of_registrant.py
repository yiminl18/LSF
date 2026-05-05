def rule_page1_phrase_number_of_shares_outstanding_of_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans with 'number of shares outstanding of the registrant' style phrasing."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "number of shares outstanding" in t and "registrant" in t:
                out.append(span)
    except Exception:
        return []
    return out
