def rule_page1_text_or_textspan_contains_our_telephone_number(doc: dict) -> list[dict]:
    """Match spans whose text or text_span says 'our telephone number'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"\bour telephone number\b", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
