def rule_page1_exchange_before_section12g(doc: dict) -> list[dict]:
    """Match spans between the Section 12(b) and Section 12(g) labels that contain exchange names."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        start = None
        stop = None
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and start is None and "section 12(b)" in low:
                start = i
            if span.get("page_no") == 1 and "section 12(g)" in low:
                stop = i
                break
        if start is None:
            return []
        if stop is None:
            stop = min(len(texts), start + 15)
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for i in range(start, stop):
            s = texts[i]
            if s.get("page_no") == 1 and ex_re.search((s.get("text") or "").lower()):
                out.append(s)
        return out
    except Exception:
        return []
