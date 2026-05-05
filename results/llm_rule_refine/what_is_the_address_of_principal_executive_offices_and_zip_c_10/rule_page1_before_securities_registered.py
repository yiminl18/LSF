def rule_page1_before_securities_registered(doc: dict) -> list[dict]:
    """Return address-like spans appearing before the first securities-registered line on page 1."""
    try:
        import re
        spans = doc.get("texts", [])
        cutoff = None
        for i, s in enumerate(spans):
            if s.get("page_no") == 1:
                txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                if "securities registered pursuant to section 12(b)" in txt:
                    cutoff = i
                    break
        if cutoff is None:
            cutoff = len(spans)
        out = []
        for s in spans[:cutoff]:
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and (
                re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low)
                or "address of principal executive offices" in low
            ):
                out.append(s)
        return out
    except Exception:
        return []
