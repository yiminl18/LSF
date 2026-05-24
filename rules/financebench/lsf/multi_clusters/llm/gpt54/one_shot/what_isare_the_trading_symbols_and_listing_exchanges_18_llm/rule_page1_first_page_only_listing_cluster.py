def rule_page1_first_page_only_listing_cluster(doc: dict) -> list[dict]:
    """Match the first-page cluster where listing answers almost always appear."""
    try:
        texts = doc.get("texts", [])
        return [s for s in texts if s.get("page_no") == 1 and s.get("label") in {"text", "section_header", "table"}]
    except Exception:
        return []
