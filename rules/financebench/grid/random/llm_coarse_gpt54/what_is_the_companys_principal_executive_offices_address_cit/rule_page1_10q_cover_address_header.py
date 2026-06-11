def rule_page1_10q_cover_address_header(doc: dict) -> list[dict]:
    """Match page-1 10-Q H2 headers that are full address lines."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\b\d+\s+[^,]+,\s*[A-Z][A-Za-z .-]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York)\s+\d{5}', txt):
                out.append(span)
            elif re.search(r'Warmley,\s*Bristol.*United Kingdom', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
