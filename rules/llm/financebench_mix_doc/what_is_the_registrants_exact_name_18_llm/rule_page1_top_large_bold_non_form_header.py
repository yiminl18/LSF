def rule_page1_top_large_bold_non_form_header(doc: dict) -> list[dict]:
    """Match top-of-page large bold non-form headers on page 1 that are likely the registrant name."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and float(span.get("size", 0) or 0) >= 12
                and span.get("label") in {"text", "section_header"}
                and "form 10-" not in low
                and "form 8-k" not in low
                and "current report" not in low
                and "securities and exchange commission" not in low
                and "united states" not in low
                and "washington, d.c." not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
