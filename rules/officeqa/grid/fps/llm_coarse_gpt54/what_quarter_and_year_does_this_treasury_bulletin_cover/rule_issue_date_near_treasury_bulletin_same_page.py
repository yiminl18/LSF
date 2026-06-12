def rule_issue_date_near_treasury_bulletin_same_page(doc: dict) -> list[dict]:
    """Match date-like spans on the same page as a Treasury Bulletin title, favoring cover-page co-location."""
    import re
    try:
        texts = doc.get("texts", [])
        tb_pages = set(
            s.get("page_no") for s in texts
            if re.search(r"treasury\s+bulletin", (s.get("text") or "").strip(), re.I)
        )
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter|"
            r"First|Second|Third|Fourth|1st|2nd|3rd|4th"
            r")\b",
            re.I,
        )
        return [
            s for s in texts
            if s.get("page_no") in tb_pages and pat.search((s.get("text") or "").strip()) and not re.search(r"treasury\s+bulletin", (s.get("text") or "").strip(), re.I)
        ]
    except Exception:
        return []
