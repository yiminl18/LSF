def rule_profile_of_economy_gdp_growth_first_two_pages(doc: dict) -> list[dict]:
    """Match GDP-growth spans in the first two pages of the Profile of the Economy article."""
    import re
    try:
        pages = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in path:
                p = span.get("page_no")
                if p is not None:
                    pages.append(p)
        if not pages:
            return []
        first_pages = set(sorted(set(pages))[:2])
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "")
            if p in first_pages and "Profile of the Economy" in path and re.search(r"GDP|gross domestic product", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
