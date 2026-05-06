def rule_page1_exact_name_in_text_span(doc: dict) -> list[dict]:
    """Match page-1 spans whose own text contains the exact-name parenthetical and starts with the registrant name."""
    try:
        texts = doc.get("texts", [])
        out = []
        marker = "(Exact name of registrant as specified in its charter)"
        for span in texts:
            txt = span.get("text") or ""
            if span.get("page_no") == 1 and marker in txt:
                prefix = txt.split(marker)[0].strip(" -–—|,;:\n\t")
                if prefix:
                    out.append(span)
        return out
    except Exception:
        return []
