def rule_page1_company_header_neighbors(doc: dict) -> list[dict]:
    """Match neighbors of the main company-name header on page 1, where cover-page security info often sits."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and s.get("label") == "section_header":
                lvl = ((s.get("structure") or {}).get("level") or "")
                if lvl == "H1" and "FORM 8-K" not in (s.get("text") or "") and "FORM 10-K" not in (s.get("text") or "") and "FORM 10-Q" not in (s.get("text") or ""):
                    for j in range(i, min(len(texts), i + 20)):
                        if texts[j].get("page_no") == 1:
                            out.append(texts[j])
        return out
    except Exception:
        return []
