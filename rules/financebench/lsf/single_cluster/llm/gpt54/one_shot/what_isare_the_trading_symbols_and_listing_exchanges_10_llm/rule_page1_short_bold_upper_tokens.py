def rule_page1_short_bold_upper_tokens(doc: dict) -> list[dict]:
    """Match short bold uppercase/alphanumeric page 1 spans that often hold ticker symbols."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if span.get("bold") == 1 and re.fullmatch(r"[A-Z0-9]{2,8}", txt):
                out.append(span)
        return out
    except Exception:
        return []
