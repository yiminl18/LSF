def rule_page1_contains_month_year_and_ended(doc: dict) -> list[dict]:
    """Match page-1 spans containing both 'ended' and a month name."""
    import re
    try:
        month = r'january|february|march|april|may|june|july|august|september|october|november|december'
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "ended" in txt and re.search(month, txt):
                out.append(span)
        return out
    except Exception:
        return []
