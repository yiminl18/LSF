def rule_profile_of_economy_early_pages_all(doc: dict) -> list[dict]:
    """Match all Profile of the Economy spans on early pages as a high-recall fallback."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page is not None and page <= 12 and "profile of the economy" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
