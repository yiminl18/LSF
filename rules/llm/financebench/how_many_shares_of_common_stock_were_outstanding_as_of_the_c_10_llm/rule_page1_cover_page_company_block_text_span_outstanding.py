def rule_page1_cover_page_company_block_text_span_outstanding(doc: dict) -> list[dict]:
    """Match company cover-page section headers whose text_span contains the outstanding-share disclosure."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            ts = (span.get("text_span") or "").lower()
            if "outstanding" in ts and ("common stock" in ts or "shares" in ts):
                out.append(span)
    except Exception:
        return []
    return out
