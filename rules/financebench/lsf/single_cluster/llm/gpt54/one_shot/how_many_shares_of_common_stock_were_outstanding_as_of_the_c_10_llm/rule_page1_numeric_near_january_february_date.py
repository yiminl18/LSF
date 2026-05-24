def rule_page1_numeric_near_january_february_date(doc: dict) -> list[dict]:
    """Match numeric spans on page 1 near January/February cover-page reference dates and outstanding-share wording."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not num_re.match(txt):
                continue
            ctx = " ".join((x.get("text") or "") for x in texts[max(0, i - 8):i]).lower()
            if ("january" in ctx or "february" in ctx or "october" in ctx) and "outstanding" in ctx:
                out.append(span)
    except Exception:
        return []
    return out
