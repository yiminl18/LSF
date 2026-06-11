def rule_page1_bold_address_line_with_city_state(doc: dict) -> list[dict]:
    """Match bold page-1 spans that look like address lines with city and state/province."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\d{2,}.*[, ]+[A-Z][a-zA-Z.\'-]+[, ]+(?:[A-Z]{2}|[A-Z][a-z]+)', txt):
                out.append(span)
        return out
    except Exception:
        return []
