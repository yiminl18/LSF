def rule_page13_bold_cover_title_form_10k(doc: dict) -> list[dict]:
    """Match short bold page-1/3 cover titles that contain FORM 10-K."""
    try:
        text_spans = []
        for s in doc.get("texts", []):
            text = " ".join((s.get("text") or "").split())
            if (s.get("page_no") or 99) <= 3 and s.get("label") in {"text", "section_header"} and s.get("bold") == 1 and "FORM 10-K" in text.upper() and len(text) <= 80:
                text_spans.append(s)
        return text_spans
    except Exception:
        return []
