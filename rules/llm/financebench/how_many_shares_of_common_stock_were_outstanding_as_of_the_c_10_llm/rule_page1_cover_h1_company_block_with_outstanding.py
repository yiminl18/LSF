def rule_page1_cover_h1_company_block_with_outstanding(doc: dict) -> list[dict]:
    """Match large company-name cover-page section_header spans that inline the outstanding-share disclosure."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            t = (span.get("text") or "").lower()
            ts = (span.get("text_span") or "").lower()
            full = t + " " + ts
            if "outstanding" in full and ("common stock" in full or "shares of common stock" in full):
                out.append(span)
    except Exception:
        return []
    return out
