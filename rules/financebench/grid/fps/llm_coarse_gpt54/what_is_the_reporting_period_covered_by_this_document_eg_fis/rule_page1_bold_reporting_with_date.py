def rule_page1_bold_reporting_with_date(doc: dict) -> list[dict]:
    """Match bold page-1 spans containing reporting keywords and a 4-digit year."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text") or ""
            low = text.lower()
            if span.get("page_no") == 1 and span.get("bold") == 1 and re.search(r'\b20\d{2}\b', text):
                if any(k in low for k in ["fiscal year ended", "quarterly period ended", "date of report", "event reported"]):
                    out.append(span)
        return out
    except Exception:
        return []
