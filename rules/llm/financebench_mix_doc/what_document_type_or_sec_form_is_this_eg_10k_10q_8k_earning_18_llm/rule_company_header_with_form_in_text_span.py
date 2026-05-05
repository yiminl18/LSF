def rule_company_header_with_form_in_text_span(doc: dict) -> list[dict]:
    """Match company-name headers whose text_span contains a form name, common in cover-page OCR merges."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").strip()
            text_span = (span.get("text_span") or "").strip()
            if re.search(r"\b(FORM\s+(10-K|10-Q|8-K)|ANNUAL REPORT ON FORM 10-K|ANNUAL REPORT ON FORM 10-Q|CURRENT REPORT)\b", text_span, re.I):
                out.append(span)
        return out
    except Exception:
        return []
