def rule_page1_known_symbol_like_spans(doc: dict) -> list[dict]:
    """Match short page-1 spans that look like ticker symbols."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(r"^[A-Z]{1,6}(?:/[0-9]{2})?$")
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and pat.match(txt):
                out.append(span)
        return out
    except Exception:
        return []
