def rule_page1_company_cover_block_bold_small_text(doc: dict) -> list[dict]:
    """Match bold small-font page-1 cover-block spans, which often include the address."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            size = span.get("size") or 0
            if span.get("bold") == 1 and 7 <= size <= 10 and span.get("label") in {"text", "section_header"}:
                out.append(span)
        return out
    except Exception:
        return []
