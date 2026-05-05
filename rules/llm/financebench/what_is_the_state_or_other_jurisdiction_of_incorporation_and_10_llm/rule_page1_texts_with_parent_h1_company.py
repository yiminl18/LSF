def rule_page1_texts_with_parent_h1_company(doc: dict) -> list[dict]:
    """Match page-1 text spans whose parent is a company-name H1 cover header."""
    try:
        texts = doc.get("texts", [])
        h1_company_ids = set()
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                txt = (span.get("text") or "").lower()
                if txt and "commission" not in txt and "form 10-k" not in txt:
                    h1_company_ids.add(idx)
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("parent_id") in h1_company_ids:
                out.append(span)
        return out
    except Exception:
        return []
