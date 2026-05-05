def rule_page1_first_25_spans_with_ein_number(doc: dict) -> list[dict]:
    """Match very early page-1 spans containing an EIN number pattern."""
    try:
        import re
        out = []
        for span in doc.get("texts", [])[:25]:
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", text):
                out.append(span)
        return out
    except Exception:
        return []
