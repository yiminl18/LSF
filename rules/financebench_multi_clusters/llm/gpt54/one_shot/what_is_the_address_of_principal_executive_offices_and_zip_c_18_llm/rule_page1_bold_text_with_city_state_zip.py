def rule_page1_bold_text_with_city_state_zip(doc: dict) -> list[dict]:
    """Match bold page-1 text spans containing city/state/ZIP style address endings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r',\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$', text) or re.search(r',\s*[A-Za-z ]+\s+\d{5}(?:-\d{4})?$', text):
                out.append(span)
        return out
    except Exception:
        return []
