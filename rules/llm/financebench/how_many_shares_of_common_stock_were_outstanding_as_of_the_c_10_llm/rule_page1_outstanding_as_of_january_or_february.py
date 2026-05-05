def rule_page1_outstanding_as_of_january_or_february(doc: dict) -> list[dict]:
    """Match page-1 spans with outstanding-share language and a cover-page reference date in January/February."""
    out = []
    try:
        for span in doc.get("texts", []):
            t = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "outstanding" in t and ("january" in t or "february" in t or "october" in t):
                if "common stock" in t or "shares" in t:
                    out.append(span)
    except Exception:
        return []
    return out
