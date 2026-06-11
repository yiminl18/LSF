def rule_page1_bold_address_line_with_zip(doc: dict) -> list[dict]:
    """Match bold page-1 body spans that look like a full address line with zip code."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            if re.search(r'\b\d{5}(?:-\d{4})?\b', txt) and re.search(r',', txt):
                out.append(span)
            elif re.search(r'\b[A-Z]{1,3}\s*\d[A-Z0-9 ]+\b', txt) and re.search(r'United Kingdom|UK|Bristol', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
