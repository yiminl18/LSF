def rule_form_page1_any_label(doc: dict) -> list[dict]:
    """Match any page-1 span containing FORM 10-K/10-Q/8-K regardless of label."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
