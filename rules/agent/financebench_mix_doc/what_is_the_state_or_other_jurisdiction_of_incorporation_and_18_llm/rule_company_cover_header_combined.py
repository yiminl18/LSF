def rule_company_cover_header_combined(doc: dict) -> list[dict]:
    """Retrieve page-1 company cover section headers that already contain both incorporation and EIN cues."""
    try:
        texts = doc.get("texts", [])
        out = []
        for s in texts:
            if s.get("page_no") != 1:
                continue
            if s.get("label") != "section_header":
                continue
            txt = (s.get("text") or "")
            span = (s.get("text_span") or "")
            combo = (txt + " " + span).lower()
            if "exact name of registrant" in combo and (
                "i.r.s. employer identification no" in combo or "irs employer identification no" in combo
            ) and (
                "state or other jurisdiction of incorporation" in combo or
                "state or other jurisdiction of incorporation or organization" in combo or
                "state or other jurisdiction of incorporation)" in combo or
                "state or other jurisdiction of incorporation or organization)" in combo
            ):
                out.append(s)
                continue
            if ("i.r.s. employer identification no" in combo or "irs employer identification no" in combo) and (
                "state or other jurisdiction of incorporation" in combo or
                "state or other jurisdiction of incorporation or organization" in combo
            ):
                out.append(s)
        return out
    except Exception:
        return []

