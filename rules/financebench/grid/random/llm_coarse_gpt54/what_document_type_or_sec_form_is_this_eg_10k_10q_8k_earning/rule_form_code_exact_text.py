def rule_form_code_exact_text(doc: dict) -> list[dict]:
    """Match spans whose text is exactly FORM 10-K, FORM 10-Q, or FORM 8-K."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.fullmatch(r"\s*FORM\s+(10-K|10-Q|8-K)\s*", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
