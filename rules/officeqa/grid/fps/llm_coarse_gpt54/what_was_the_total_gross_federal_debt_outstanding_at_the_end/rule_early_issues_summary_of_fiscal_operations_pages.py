def rule_early_issues_summary_of_fiscal_operations_pages(doc: dict) -> list[dict]:
    """Match tables on early pages 15-19 under Summary of Fiscal Operations, common in older issues."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if page is not None and 15 <= page <= 19:
                if re.search(r'summary of fiscal operations|ffo[\-–— ]?1', path + " " + text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
