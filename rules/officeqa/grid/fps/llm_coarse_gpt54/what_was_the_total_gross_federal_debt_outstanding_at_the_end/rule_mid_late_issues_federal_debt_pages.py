def rule_mid_late_issues_federal_debt_pages(doc: dict) -> list[dict]:
    """Match Federal Debt tables on mid-document pages 17-35, common in later quarterly issues."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            page = span.get("page_no")
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if page is not None and 17 <= page <= 35:
                if re.search(r'federal debt|fd[\-–— ]?1|summary of federal debt', path + " " + text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
