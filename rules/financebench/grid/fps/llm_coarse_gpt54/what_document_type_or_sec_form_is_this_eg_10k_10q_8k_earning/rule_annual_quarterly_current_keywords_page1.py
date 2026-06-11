def rule_annual_quarterly_current_keywords_page1(doc: dict) -> list[dict]:
    """Match page-1 spans containing annual, quarterly, or current report phrases."""
    import re
    try:
        out = []
        pat = r"(ANNUAL REPORT PURSUANT|QUARTERLY REPORT PURSUANT|CURRENT REPORT)"
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(pat, text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
