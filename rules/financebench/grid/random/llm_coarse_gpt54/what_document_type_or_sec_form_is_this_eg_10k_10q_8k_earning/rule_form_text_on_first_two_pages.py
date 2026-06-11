def rule_form_text_on_first_two_pages(doc: dict) -> list[dict]:
    """Match form-code spans on pages 1-2, where the document type almost always appears."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") in {1, 2}
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
