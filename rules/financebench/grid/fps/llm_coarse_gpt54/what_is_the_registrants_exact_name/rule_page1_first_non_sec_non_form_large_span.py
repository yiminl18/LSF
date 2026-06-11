def rule_page1_first_non_sec_non_form_large_span(doc: dict) -> list[dict]:
    """Match the first large page-1 span that is not SEC or form boilerplate."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if float(span.get("size") or 0) < 10:
                continue
            if "securities and exchange commission" in txt or "washington" in txt or "form 10-" in txt or "form 8-k" in txt or "current report" in txt or "annual report" in txt or "quarterly report" in txt:
                continue
            out.append(span)
            break
        return out
    except Exception:
        return []
