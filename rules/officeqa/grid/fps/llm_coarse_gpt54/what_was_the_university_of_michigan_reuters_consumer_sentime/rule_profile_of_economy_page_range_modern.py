def rule_profile_of_economy_page_range_modern(doc: dict) -> list[dict]:
    """Match modern Profile of the Economy spans on early content pages where macro indicator readings usually appear."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if page is not None and 7 <= page <= 12 and "profile of the economy" in path.lower():
                if re.search(r"(consumer|sentiment|confidence|michigan|reuters)", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
