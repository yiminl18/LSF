def rule_page1_text_or_textspan_contains_registrant_phone(doc: dict) -> list[dict]:
    """Match page-1 spans whose text or text_span contains the registrant phone phrase."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if span.get("page_no") == 1 and re.search(r"registrant[’'`s]{0,2}\s+telephone number", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
