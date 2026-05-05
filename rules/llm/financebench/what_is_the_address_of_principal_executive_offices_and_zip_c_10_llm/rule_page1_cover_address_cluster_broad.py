def rule_page1_cover_address_cluster_broad(doc: dict) -> list[dict]:
    """Broadly match the page-1 cover cluster containing company name, address, ZIP, and phone."""
    try:
        spans = doc.get("texts", [])
        out = []
        for s in spans:
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if any(k in txt for k in [
                "exact name of registrant",
                "state or other jurisdiction",
                "state of incorporation",
                "address of principal executive offices",
                "address and telephone number, including area code, of registrant",
                "address of principal executive offices and zip code",
                "zip code",
                "telephone number"
            ]):
                out.append(s)
        return out
    except Exception:
        return []
