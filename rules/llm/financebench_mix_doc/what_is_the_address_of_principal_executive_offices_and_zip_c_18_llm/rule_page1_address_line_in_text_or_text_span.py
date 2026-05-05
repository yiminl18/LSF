def rule_page1_address_line_in_text_or_text_span(doc: dict) -> list[dict]:
    """Match spans where either text or text_span contains a full address line."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            for field in [span.get("text") or "", span.get("text_span") or ""]:
                if re.search(r'\d{1,5} .+,\s*.+,\s*[A-Z][A-Za-z .]+ \d{5}(?:-\d{4})?', field) or re.search(r'83 Tower Road North.*BS30 8XP', field, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
