def rule_form_type_cover_page_section_headers(doc: dict) -> list[dict]:
    """Match page-1 section headers that are likely the document type answer."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1 or s.get("label") != "section_header":
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).upper()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I) or "CURRENT REPORT" in txt:
                out.append(s)
        return out
    except Exception:
        return []
