def rule_issue_month_on_cover_pages_1_to_3(doc: dict) -> list[dict]:
    """Match issue date spans on pages 1-3 where later bulletins place the cover and contents."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December"
            r")\s+\d{4}\b|"
            r"\b(Spring|Summer|Fall|Winter)\s+Issue(?:\s+December\s+\d{4})?\b|"
            r"\b(First|Second|Third|Fourth)\s+Quarter,?\s+Fiscal\s+\d{4}\b",
            re.I,
        )
        return [s for s in texts if 1 <= s.get("page_no", 999) <= 3 and pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
