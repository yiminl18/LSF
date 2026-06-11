def rule_cover_page_company_header_with_embedded_table(doc: dict) -> list[dict]:
    """Match large company-name cover spans whose text_span embeds the securities registration table."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") == 1
            and s.get("label") == "section_header"
            and re.search(r"Securities registered pursuant to Section 12\(b\)", (s.get("text_span") or ""), re.I)
        ]
    except Exception:
        return []
