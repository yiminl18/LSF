def rule_form_8k_h1(doc: dict) -> list[dict]:
    """Match H1/section_header spans with FORM 8-K."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r"\bFORM\s+8-K\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
