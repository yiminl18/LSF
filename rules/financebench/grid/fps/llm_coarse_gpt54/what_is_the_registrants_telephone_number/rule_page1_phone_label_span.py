def rule_page1_phone_label_span(doc: dict) -> list[dict]:
    """Match page-1 spans containing the registrant telephone label phrase."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
                if re.search(r"registrant[’'`s]{0,2}\s+telephone\s+number", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
