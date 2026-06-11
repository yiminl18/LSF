def rule_page1_company_header_text_span_period(doc: dict) -> list[dict]:
    """Match company-name H1 headers whose text_span includes a repeated period phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            head = span.get("text") or ""
            tail = span.get("text_span") or ""
            if re.search(r'(INC\.|CORPORATION|PLC|COMPANY)', head, re.I):
                if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|event reported)', tail, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
