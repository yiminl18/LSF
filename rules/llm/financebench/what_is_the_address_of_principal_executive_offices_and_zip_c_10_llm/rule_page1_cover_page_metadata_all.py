def rule_page1_cover_page_metadata_all(doc: dict) -> list[dict]:
    """Return all page-1 metadata-like spans around the registrant identity block for high recall."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if any(k in txt for k in [
                "exact name of registrant",
                "state or other jurisdiction",
                "state of incorporation",
                "employer identification no",
                "address of principal executive offices",
                "address and telephone number, including area code, of registrant",
                "zip code",
                "registrant’s telephone number",
                "registrant's telephone number",
            ]):
                out.append(span)
        return out
    except Exception:
        return []
