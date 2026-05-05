def rule_page1_text_span_with_shares_and_date(doc: dict) -> list[dict]:
    """Match page-1 text spans containing 'shares' and a month-day-year date in the cover block."""
    import re
    try:
        out = []
        months = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        for span in doc.get("texts", []):
            if span.get("page_no") not in (1, 2):
                continue
            t = " ".join((span.get("text") or "").lower().split())
            if "shares" in t and re.search(months + r"\s+\d{1,2},\s+\d{4}", t):
                out.append(span)
        return out
    except Exception:
        return []
