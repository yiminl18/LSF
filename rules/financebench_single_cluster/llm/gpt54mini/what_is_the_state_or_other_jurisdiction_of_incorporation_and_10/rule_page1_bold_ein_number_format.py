def rule_page1_bold_ein_number_format(doc: dict) -> list[dict]:
    """Match bold page-1 spans that look like standalone EIN numbers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("bold") == 1 and re.fullmatch(r"\d{2}-\d{7}", text):
                out.append(span)
        return out
    except Exception:
        return []
