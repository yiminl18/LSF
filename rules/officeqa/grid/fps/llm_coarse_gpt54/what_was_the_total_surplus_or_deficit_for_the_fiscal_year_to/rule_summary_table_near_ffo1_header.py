def rule_summary_table_near_ffo1_header(doc: dict) -> list[dict]:
    """Match tables appearing on the same page as a header mentioning FFO-1 or Summary of Fiscal Operations."""
    import re
    out = []
    try:
        pages = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r'(table\s+)?FFO[-\s]?1', txt, re.I) or re.search(r'summary of fiscal operations', txt, re.I):
                pages.add(span.get("page_no"))
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in pages:
                out.append(span)
    except Exception:
        return []
    return out
