def rule_esf_table_on_header_page_plus_one(doc: dict) -> list[dict]:
    """Match tables on the page of the ESF header or the next page."""
    import re
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header" and re.search(r'EXCHANGE STABILIZATION FUND', span.get("text", ""), re.I):
                p = span.get("page_no")
                if isinstance(p, int):
                    pages.add(p)
                    pages.add(p + 1)
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table" and s.get("page_no") in pages
        ]
    except Exception:
        return []
