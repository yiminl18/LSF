def rule_form_code_with_exchange_act_reference(doc: dict) -> list[dict]:
    """Match spans containing a form code and 'Securities Exchange Act of 1934' reference."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I)
            and re.search(r"Securities Exchange Act of 1934", span.get("text") or "", re.I)
        ]
    except Exception:
        return []
