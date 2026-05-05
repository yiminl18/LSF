def rule_page1_text_span_period_phrases(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text_span contains the reporting period phrase."""
    import re
    try:
        pats = [
            r'for the fiscal year ended',
            r'for the fiscal year end(?:ed)?',
            r'for the quarterly period ended',
            r'for the quarter ended',
            r'date of report \(date of earliest event reported\)',
        ]
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text_span") or "").lower()
            if span.get("page_no") == 1 and txt and any(re.search(p, txt) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
