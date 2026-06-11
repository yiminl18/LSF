def rule_page1_top_large_bold_non_form(doc: dict) -> list[dict]:
    """Match top-of-page-1 large bold spans excluding SEC and form boilerplate."""
    try:
        out = []
        for span in doc.get("texts", [])[:25]:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and float(span.get("size") or 0) >= 10
                and txt
                and "form 8-k" not in low
                and "form 10-k" not in low
                and "form 10-q" not in low
                and "securities and exchange commission" not in low
                and "united states" not in low
                and "current report" not in low
            ):
                out.append(span)
        return out
    except Exception:
        return []
