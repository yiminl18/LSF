def rule_page_after_statement_of_income_header(doc: dict) -> list[dict]:
    """Match spans on the same or next page after a Statement of Income/Operations header."""
    import re
    try:
        pages = set()
        for s in doc.get("texts", []):
            if s.get("label") == "section_header" and re.search(r"(statement|statements) of (income|operations|earnings)", s.get("text", "") or "", re.I):
                p = s.get("page_no")
                if isinstance(p, int):
                    pages.add(p)
                    pages.add(p + 1)
        if not pages:
            return []
        return [s for s in doc.get("texts", []) if s.get("page_no") in pages]
    except Exception:
        return []
