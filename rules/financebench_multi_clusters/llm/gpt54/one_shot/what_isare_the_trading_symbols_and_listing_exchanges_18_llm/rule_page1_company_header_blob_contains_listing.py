def rule_page1_company_header_blob_contains_listing(doc: dict) -> list[dict]:
    """Match large page-1 company header spans whose text_span contains listing information inline."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            txt = ((span.get("text_span") or "") + " " + (span.get("text") or "")).lower()
            if (
                "securities registered pursuant to section 12(b)" in txt
                or "trading symbol" in txt
                or "name of each exchange on which registered" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
