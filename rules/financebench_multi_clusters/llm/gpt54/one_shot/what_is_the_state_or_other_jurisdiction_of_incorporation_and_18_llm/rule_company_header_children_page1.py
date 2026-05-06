def rule_company_header_children_page1(doc: dict) -> list[dict]:
    """Match body children of the page-1 company-name H1 header."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_h1_ids = []
        for idx, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text", "") or "").upper()
                and "SECURITIES AND EXCHANGE COMMISSION" not in (span.get("text", "") or "").upper()
            ):
                company_h1_ids.append(idx)
        for pid in company_h1_ids:
            for span in texts:
                if span.get("page_no") == 1 and span.get("structure", {}).get("parent_id") == pid:
                    out.append(span)
        return out
    except Exception:
        return []
