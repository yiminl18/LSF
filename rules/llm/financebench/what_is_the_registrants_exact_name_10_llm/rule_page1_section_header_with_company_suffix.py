def rule_page1_section_header_with_company_suffix(doc: dict) -> list[dict]:
    """Match page-1 headings containing common company suffixes like Inc., Corporation, Company, plc, Incorporated."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r"\b(inc\.?|corporation|company|plc|incorporated)\b", re.I)
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
