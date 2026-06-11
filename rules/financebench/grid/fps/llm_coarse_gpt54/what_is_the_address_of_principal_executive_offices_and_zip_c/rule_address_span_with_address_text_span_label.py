def rule_address_span_with_address_text_span_label(doc: dict) -> list[dict]:
    """Match spans whose text_span explicitly contains '(Address of principal executive offices)'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            ts = span.get("text_span") or ""
            if re.search(r'address of principal executive offices', ts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
