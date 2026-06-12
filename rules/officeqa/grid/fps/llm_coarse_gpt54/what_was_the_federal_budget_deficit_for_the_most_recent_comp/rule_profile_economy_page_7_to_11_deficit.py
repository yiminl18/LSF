def rule_profile_economy_page_7_to_11_deficit(doc: dict) -> list[dict]:
    """Match likely answer spans on early Profile of the Economy pages where deficit percent is discussed."""
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                isinstance(page, int) and 5 <= page <= 11
                and span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit" in low
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
