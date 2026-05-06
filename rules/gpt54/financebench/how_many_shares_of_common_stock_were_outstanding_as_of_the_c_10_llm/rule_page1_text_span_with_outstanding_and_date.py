def rule_page1_text_span_with_outstanding_and_date(doc: dict) -> list[dict]:
    """Match page-1 text spans containing both 'outstanding' and a month-day-year date."""
    import re
    try:
        out = []
        months = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        for span in doc.get("texts", []):
            if span.get("page_no") not in (1, 2):
                continue
            if span.get("label") not in ("text", "section_header"):
                continue
            t = " ".join((span.get("text") or "").lower().split())
            if "outstanding" in t and re.search(months + r"\s+\d{1,2},\s+\d{4}", t):
                out.append(span)
        return out
    except Exception:
        return []
