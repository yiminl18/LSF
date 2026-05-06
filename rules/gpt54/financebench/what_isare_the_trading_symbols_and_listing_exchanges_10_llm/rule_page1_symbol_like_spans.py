def rule_page1_symbol_like_spans(doc: dict) -> list[dict]:
    """Match short page 1 spans that look like ticker symbols."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"[A-Z]{1,6}[0-9A-Z]{0,4}", txt):
                out.append(span)
        return out
    except Exception:
        return []
