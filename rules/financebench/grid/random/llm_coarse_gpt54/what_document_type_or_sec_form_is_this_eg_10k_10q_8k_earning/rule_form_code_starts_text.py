def rule_form_code_starts_text(doc: dict) -> list[dict]:
    """Match spans whose text starts with FORM 10-K/10-Q/8-K."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.match(r"\s*FORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
