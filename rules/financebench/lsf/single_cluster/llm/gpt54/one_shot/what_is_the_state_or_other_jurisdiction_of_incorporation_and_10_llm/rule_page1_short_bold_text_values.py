def rule_page1_short_bold_text_values(doc: dict) -> list[dict]:
    """Match short bold page-1 text spans that are likely cover-page metadata values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "text" and span.get("bold") == 1:
                if len(text.split()) <= 6 and (re.fullmatch(r"\d{2}-\d{7}", text) or not re.search(r"^\(?[A-Za-z ]{20,}\)?$", text)):
                    out.append(span)
        return out
    except Exception:
        return []
