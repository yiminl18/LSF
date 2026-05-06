def rule_page1_top_headers_with_period_in_text_or_span(doc: dict) -> list[dict]:
    """Match top page-1 headers where either text or text_span contains the reporting period."""
    import re
    try:
        out = []
        for span in doc.get("texts", [])[:25]:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if re.search(r'(fiscal year ended|quarterly period ended|date of report|earliest event reported)', txt):
                out.append(span)
        return out
    except Exception:
        return []
