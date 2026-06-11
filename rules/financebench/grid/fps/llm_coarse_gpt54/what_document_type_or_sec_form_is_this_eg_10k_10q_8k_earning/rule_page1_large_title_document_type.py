def rule_page1_large_title_document_type(doc: dict) -> list[dict]:
    """Match large page-1 titles that are likely the document type or release type."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and float(span.get("size") or 0) >= 11
                and re.search(r"(FORM\s+10-K|FORM\s+10-Q|FORM\s+8-K|NEWS RELEASE|CURRENT REPORT)", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
