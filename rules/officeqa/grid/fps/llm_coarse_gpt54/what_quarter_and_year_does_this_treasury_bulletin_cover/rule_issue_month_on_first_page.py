def rule_issue_month_on_first_page(doc: dict) -> list[dict]:
    """Match issue month/date spans on page 1, common in modern bulletins."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December"
            r")\s+\d{4}\b|"
            r"\b(Spring|Summer|Fall|Winter)\s+Issue\b",
            re.I,
        )
        return [s for s in texts if s.get("page_no") == 1 and pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
