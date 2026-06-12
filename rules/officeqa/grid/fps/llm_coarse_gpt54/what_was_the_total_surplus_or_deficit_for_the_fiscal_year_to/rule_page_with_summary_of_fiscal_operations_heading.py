def rule_page_with_summary_of_fiscal_operations_heading(doc: dict) -> list[dict]:
    """Match all table spans on pages containing a heading 'Summary of Fiscal Operations'."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            if re.search(r'summary of fiscal operations', span.get("text", ""), re.I):
                pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
