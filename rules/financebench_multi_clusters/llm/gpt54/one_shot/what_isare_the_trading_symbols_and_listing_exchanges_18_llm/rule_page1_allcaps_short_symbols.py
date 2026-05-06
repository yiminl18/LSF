def rule_page1_allcaps_short_symbols(doc: dict) -> list[dict]:
    """Match short all-caps page-1 spans likely to be ticker symbols."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if len(txt) <= 8 and re.fullmatch(r"[A-Z0-9/]{1,8}", txt):
                if any(ch.isalpha() for ch in txt):
                    out.append(span)
        return out
    except Exception:
        return []
