def rule_page1_2_section12b_title_spans_without_table(doc: dict) -> list[dict]:
    """Match short page 1-2 class-title spans in a Section 12(b) block when no table is present."""
    try:
        texts = doc.get("texts", [])
        has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
            for s in texts
        )
        if has_table:
            return []

        anchor_pages = {
            s.get("page_no")
            for s in texts
            if s.get("page_no", 999) <= 2
            and "securities registered pursuant to section 12(b)"
            in s.get("text", "").lower()
        }
        if not anchor_pages:
            return []

        keys = (
            "common stock",
            "ordinary share",
            "ordinary shares",
            "class a common stock",
            "title of each class",
            "notes due",
            "depositary instrument",
            "par value",
        )
        return [
            s for s in texts
            if s.get("page_no") in anchor_pages
            and len(s.get("text", "").lower().split()) <= 14
            and any(k in s.get("text", "").lower() for k in keys)
        ]
    except Exception:
        return []
