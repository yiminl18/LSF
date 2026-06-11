def rule_page1_top_bold_form_family(doc: dict) -> list[dict]:
    """Match bold page-1 spans near the top that contain FORM, REPORT, or RELEASE keywords."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts[:25]):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and re.search(r"\b(FORM|REPORT|RELEASE)\b", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
