def rule_page1_period_in_large_bold_headers(doc: dict) -> list[dict]:
    """Match large bold page-1 headers that embed the reporting period in text_span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            size = span.get("size") or 0
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and span.get("bold") == 1 and size >= 10:
                if re.search(r'(fiscal year ended|quarterly period ended|date of report)', txt):
                    out.append(span)
        return out
    except Exception:
        return []
