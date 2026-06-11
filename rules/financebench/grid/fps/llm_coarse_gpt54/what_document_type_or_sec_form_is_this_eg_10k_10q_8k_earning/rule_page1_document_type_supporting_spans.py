def rule_page1_document_type_supporting_spans(doc: dict) -> list[dict]:
    """Return page-1 spans that support document type identification via form/report/release keywords."""
    import re
    try:
        out = []
        pat = r"(FORM\s+10-K|FORM\s+10-Q|FORM\s+8-K|ANNUAL REPORT|QUARTERLY REPORT|CURRENT REPORT|NEWS RELEASE)"
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and re.search(pat, (span.get("text") or ""), re.I):
                out.append(span)
        return out
    except Exception:
        return []
