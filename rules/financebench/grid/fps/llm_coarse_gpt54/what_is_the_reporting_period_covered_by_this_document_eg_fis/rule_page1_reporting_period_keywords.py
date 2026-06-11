def rule_page1_reporting_period_keywords(doc: dict) -> list[dict]:
    """Match page-1 spans containing standard reporting-period phrases like fiscal year, quarterly period, or date of report."""
    import re
    try:
        out = []
        pats = [
            r'for the fiscal year ended',
            r'fiscal year ended',
            r'for the quarterly period ended',
            r'quarter ended',
            r'quarterly period ended',
            r'date of report',
            r'date of earliest event reported',
            r'event reported',
            r'current report',
        ]
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and any(re.search(p, text) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
