def rule_profile_of_economy_first_pages_unemployment(doc: dict) -> list[dict]:
    """Match unemployment-related text on early pages where Profile of the Economy usually appears."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            text = span.get("text", "") or ""
            if span.get("label") == "text" and isinstance(page, int) and 5 <= page <= 11:
                if re.search(r"\bunemployment rate\b", text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
