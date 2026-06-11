def rule_page1_h1_company_name_not_form(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers that are not form/report/SEC boilerplate."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") != "section_header":
                continue
            if span.get("structure", {}).get("level") != "H1":
                continue
            txt = (span.get("text", "") or "").strip().lower()
            if any(x in txt for x in ["form 10-k", "form 10-q", "form 8-k", "current report", "securities and exchange commission", "united states", "documents incorporated by reference", "part i", "news release", "or"]):
                continue
            out.append(span)
        return out
    except Exception:
        return []
