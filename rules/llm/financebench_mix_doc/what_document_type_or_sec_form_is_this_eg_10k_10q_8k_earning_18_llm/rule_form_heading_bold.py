def rule_form_heading_bold(doc: dict) -> list[dict]:
    """Match bold spans containing a form heading."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("bold") == 1 and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
