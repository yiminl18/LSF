def rule_cover_page_company_section_header(doc: dict) -> list[dict]:
    """Match the large company-name section header on page 1 that often embeds the 12(b) registration text in text_span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "securities registered pursuant to section 12(b)" in txt:
                out.append(span)
        return out
    except Exception:
        return []
