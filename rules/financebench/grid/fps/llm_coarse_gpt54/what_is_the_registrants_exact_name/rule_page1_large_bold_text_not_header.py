def rule_page1_large_bold_text_not_header(doc: dict) -> list[dict]:
    """Match large bold page-1 text spans (not only section headers) that often hold the registrant name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if float(span.get("size") or 0) < 14:
                continue
            if "form 10-" in txt or "form 8-k" in txt or "securities and exchange commission" in txt:
                continue
            out.append(span)
        return out
    except Exception:
        return []
