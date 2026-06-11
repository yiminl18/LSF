def rule_inline_section12b_company_header(doc: dict) -> list[dict]:
    """Match page-1 company header spans whose inline text contains the Section 12(b) registration block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = ((span.get("text_span") or "") + " " + (span.get("text") or "")).lower()
                if "securities registered pursuant to section 12(b)" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
