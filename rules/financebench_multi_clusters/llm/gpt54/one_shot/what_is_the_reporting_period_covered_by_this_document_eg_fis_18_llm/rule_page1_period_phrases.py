def rule_page1_period_phrases(doc: dict) -> list[dict]:
    """Match page-1 spans containing common reporting-period lead phrases."""
    import re
    try:
        pats = [
            r'for the fiscal year ended',
            r'for the fiscal year end(?:ed)?',
            r'for the quarterly period ended',
            r'for the quarter ended',
            r'for the period ended',
            r'date of report \(date of earliest event reported\)',
            r'for the transition period from',
        ]
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and any(re.search(p, txt) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
