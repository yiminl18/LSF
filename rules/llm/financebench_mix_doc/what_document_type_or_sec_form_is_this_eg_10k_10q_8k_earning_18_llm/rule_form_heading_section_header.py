def rule_form_heading_section_header(doc: dict) -> list[dict]:
    """Match section_header spans that are SEC form headings."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or "").strip(), re.I)
        ]
    except Exception:
        return []
