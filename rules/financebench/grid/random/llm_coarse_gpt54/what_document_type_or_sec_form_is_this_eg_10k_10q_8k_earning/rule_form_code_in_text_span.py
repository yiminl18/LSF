def rule_form_code_in_text_span(doc: dict) -> list[dict]:
    """Match any span whose text_span contains FORM 10-K/10-Q/8-K."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text_span") or "", re.I)
        ]
    except Exception:
        return []
