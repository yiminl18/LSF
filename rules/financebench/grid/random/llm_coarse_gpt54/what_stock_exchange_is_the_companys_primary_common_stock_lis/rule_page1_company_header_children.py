def rule_page1_company_header_children(doc: dict) -> list[dict]:
    """Return body children under the main company H1 on page 1, where exchange listing details usually appear."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_h1_ids = []
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = ((span.get("structure") or {}).get("level") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                txt = (span.get("text") or "")
                if lvl == "H1" and txt and "form 10-" not in txt.lower() and "securities and exchange commission" not in txt.lower():
                    company_h1_ids.append(idx)
        for pid in company_h1_ids:
            for span in texts:
                st = span.get("structure") or {}
                if span.get("page_no") == 1 and st.get("parent_id") == pid:
                    out.append(span)
        return out
    except Exception:
        return []
