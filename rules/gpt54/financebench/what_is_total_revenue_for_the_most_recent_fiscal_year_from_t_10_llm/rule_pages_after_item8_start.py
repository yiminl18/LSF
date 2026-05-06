def rule_pages_after_item8_start(doc: dict) -> list[dict]:
    """Match spans on or after the first page where Item 8 begins."""
    try:
        start_page = None
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            if span.get("label") == "section_header" and "item 8" in txt and "financial statements" in txt:
                start_page = span.get("page_no")
                break
            if "item 8" in path and "financial statements" in path:
                start_page = span.get("page_no")
                break
        if start_page is None:
            return []
        return [s for s in doc.get("texts", []) if (s.get("page_no") or 0) >= start_page]
    except Exception:
        return []
