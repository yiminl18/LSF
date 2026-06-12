def rule_esf_heading_same_page_table(doc: dict) -> list[dict]:
    """Match table spans on the same page as an Exchange Stabilization Fund header."""
    import re
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header" and re.search(r'EXCHANGE STABILIZATION FUND', span.get("text", ""), re.I):
                pages.add(span.get("page_no"))
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "table" and span.get("page_no") in pages
        ]
    except Exception:
        return []
