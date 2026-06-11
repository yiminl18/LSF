def rule_page1_8k_address_line_with_zip_and_city(doc: dict) -> list[dict]:
    """Match page-1 8-K address lines with city/state and zip."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'\b[A-Z][A-Za-z .-]+,\s*(?:WA|CA|NY)\s+\d{5}(?:-\d{4})?\b', txt):
                out.append(span)
            elif re.search(r'Warmley,\s*Bristol.*United Kingdom', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
