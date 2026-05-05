def rule_page1_sentence_with_number_of_shares_of_registrants_common_stock(doc: dict) -> list[dict]:
    """Match page-1 spans using 'The number of shares of the registrant's common stock...' wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "the number of shares" in t and "registrant" in t and "common stock" in t:
                out.append(span)
    except Exception:
        return []
    return out
