def rule_company_header_span_with_embedded_12b(doc: dict) -> list[dict]:
    """Match large company-name section headers on page 1 whose text_span embeds the Section 12(b) listing block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            blob = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if re.search(r"securities registered pursuant to section 12\(b\) of the act", blob, re.I):
                out.append(span)
        return out
    except Exception:
        return []
