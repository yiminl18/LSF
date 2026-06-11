def rule_page1_phone_with_cover_typography(doc: dict) -> list[dict]:
    """Match page-1 phone-like spans with common cover-page typography (bold or section_header or larger font)."""
    try:
        import re
        out = []
        phone_pat = re.compile(r"(\+\d[\d\-\s\(\)]{6,}\d|\(\d{3,4}\)\s*\d{3,4}[-\s]?\d{4}|\d{3}[-\s]?\d{3}[-\s]?\d{4}|\d{10,12})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if phone_pat.search(txt):
                if span.get("bold") == 1 or span.get("label") == "section_header" or float(span.get("size") or 0) >= 8:
                    out.append(span)
        return out
    except Exception:
        return []
