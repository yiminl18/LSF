def rule_page1_contains_address_and_telephone_phrase(doc: dict) -> list[dict]:
    """Match any page-1 span containing the phrase address and telephone number of registrant’s principal executive offices."""
    try:
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if "address and telephone number" in txt and "principal executive offices" in txt:
                out.append(s)
        return out
    except Exception:
        return []
