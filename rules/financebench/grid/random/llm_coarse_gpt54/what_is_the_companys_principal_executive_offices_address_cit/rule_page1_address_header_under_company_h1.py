def rule_page1_address_header_under_company_h1(doc: dict) -> list[dict]:
    """Match page-1 section headers under the company H1 that look like street-address headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if "|" in path and re.search(r'\d', txt) and not re.search(r'commission|form 10-|securities and exchange|current report|annual report|quarterly report', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
