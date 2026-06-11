def rule_page1_top_half_non_form_spans(doc: dict) -> list[dict]:
    """Match page-1 non-FORM spans in the top half of the filing cover page."""
    try:
        out = []
        for span in doc.get("texts", [])[:60]:
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and "FORM 10-" not in txt and "FORM 8-K" not in txt and "FORM 10-Q" not in txt:
                out.append(span)
        return out
    except Exception:
        return []
