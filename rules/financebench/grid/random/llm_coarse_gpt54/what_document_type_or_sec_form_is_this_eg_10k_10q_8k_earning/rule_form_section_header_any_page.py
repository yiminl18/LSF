def rule_form_section_header_any_page(doc: dict) -> list[dict]:
    """Match section headers containing FORM 10-K/10-Q/8-K on any page."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
