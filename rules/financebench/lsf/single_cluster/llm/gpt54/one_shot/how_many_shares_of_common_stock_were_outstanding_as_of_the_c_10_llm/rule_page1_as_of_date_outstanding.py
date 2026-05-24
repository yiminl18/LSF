def rule_page1_as_of_date_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans that mention a date and 'outstanding' together."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if span.get("page_no") in (1, 2):
                t = " ".join(text.lower().split())
                if "outstanding" in t and re.search(r"as of [a-z]+\s+\d{1,2},\s+\d{4}", t):
                    out.append(span)
        return out
    except Exception:
        return []
