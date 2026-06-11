def rule_page1_period_phrases(doc: dict) -> list[dict]:
    """Match page-1 spans containing common reporting-period phrases like fiscal year, quarterly period, or date of report."""
    import re
    try:
        out = []
        pats = [
            r'\bfiscal year ended\b',
            r'\bfor the fiscal year ended\b',
            r'\bfor the fiscal year\b',
            r'\bquarterly period ended\b',
            r'\bfor the quarterly period ended\b',
            r'\bdate of report\b',
            r'\bdate of earliest event reported\b',
            r'\btransition period from\b',
        ]
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and any(re.search(p, txt, re.I) for p in pats):
                out.append(span)
        return out
    except Exception:
        return []
