def rule_page1_bold_text_company_name_not_header(doc: dict) -> list[dict]:
    """Match page-1 bold text spans (not necessarily headers) that look like the registrant name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r'\b(inc\.?|incorporated|corporation|company|plc|co\.)\b', re.I)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("bold", 0) != 1:
                continue
            txt = (span.get("text", "") or "").strip()
            if not txt:
                continue
            if pat.search(txt) and "form 10-" not in txt.lower() and "form 8-k" not in txt.lower():
                out.append(span)
        return out
    except Exception:
        return []
