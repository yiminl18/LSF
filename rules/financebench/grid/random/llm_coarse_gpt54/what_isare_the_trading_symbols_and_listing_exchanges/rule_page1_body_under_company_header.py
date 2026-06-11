def rule_page1_body_under_company_header(doc: dict) -> list[dict]:
    """Return page-1 body spans under the main company H1, where the listing block usually appears."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_parent_ids = set()
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = (((span.get("structure") or {}).get("level")) or "")
                txt = span.get("text", "") or ""
                if lvl == "H1" and txt and "FORM" not in txt.upper() and "SECURITIES AND EXCHANGE COMMISSION" not in txt.upper():
                    company_parent_ids.add(i)
        for span in texts:
            if span.get("page_no") == 1 and ((span.get("structure") or {}).get("parent_id") in company_parent_ids):
                out.append(span)
        return out
    except Exception:
        return []
