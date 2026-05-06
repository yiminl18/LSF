def rule_form_heading_large_font(doc: dict) -> list[dict]:
    """Match larger-font spans containing a form heading."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if float(span.get("size") or 0) >= 10 and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
