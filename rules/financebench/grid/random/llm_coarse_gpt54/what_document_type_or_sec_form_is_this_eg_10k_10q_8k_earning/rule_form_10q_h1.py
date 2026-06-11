def rule_form_10q_h1(doc: dict) -> list[dict]:
    """Match H1/section_header spans with FORM 10-Q."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("label") == "section_header"
            and re.search(r"\bFORM\s+10-Q\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
