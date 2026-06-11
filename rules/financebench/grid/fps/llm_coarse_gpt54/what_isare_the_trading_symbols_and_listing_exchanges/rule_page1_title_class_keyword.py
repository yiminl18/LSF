def rule_page1_title_class_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Title of each class' near the cover-page securities section."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1 and re.search(r"Title of each class", (s.get("text") or ""), re.I)
        ]
    except Exception:
        return []
