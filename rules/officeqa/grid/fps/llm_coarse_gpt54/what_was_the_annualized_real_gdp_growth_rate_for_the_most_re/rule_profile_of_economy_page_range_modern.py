def rule_profile_of_economy_page_range_modern(doc: dict) -> list[dict]:
    """Match all spans on pages 5-10 that are under Profile of the Economy for broad recall."""
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page is not None and 5 <= page <= 10 and "Profile of the Economy" in path:
                out.append(span)
        return out
    except Exception:
        return []
