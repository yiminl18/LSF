def rule_profile_of_economy_first_pages(doc: dict) -> list[dict]:
    """Match all Profile of the Economy spans on the first few content pages for high recall."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page is not None and page <= 12 and "Profile of the Economy" in path:
                out.append(span)
        return out
    except Exception:
        return []
