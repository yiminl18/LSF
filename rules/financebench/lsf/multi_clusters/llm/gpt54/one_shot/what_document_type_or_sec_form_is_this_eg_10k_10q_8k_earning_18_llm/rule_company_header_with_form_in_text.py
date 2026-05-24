def rule_company_header_with_form_in_text(doc: dict) -> list[dict]:
    """Match company-name cover headers whose merged text contains a form name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "").strip()
            if re.search(r"\b(FORM\s+(10-K|10-Q|8-K)|CURRENT REPORT)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
