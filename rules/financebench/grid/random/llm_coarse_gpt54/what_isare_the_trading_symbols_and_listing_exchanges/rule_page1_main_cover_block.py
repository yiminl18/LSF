def rule_page1_main_cover_block(doc: dict) -> list[dict]:
    """Return all page-1 spans from the company H1 through the start of boilerplate checkboxes, covering the listing area."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                lvl = ((span.get("structure") or {}).get("level") or "")
                txt = span.get("text", "") or ""
                if lvl == "H1" and txt and "FORM" not in txt.upper() and "SECURITIES AND EXCHANGE COMMISSION" not in txt.upper():
                    start = i
                    break
        if start is None:
            return []
        for i in range(start, len(texts)):
            txt = texts[i].get("text", "") or ""
            if texts[i].get("page_no") != 1:
                break
            if re.search(r"indicate by check mark if the registrant is a well-known seasoned issuer", txt, re.I):
                end = i
                break
        if end is None:
            end = min(len(texts), start + 30)
        return [texts[i] for i in range(start, end) if texts[i].get("page_no") == 1]
    except Exception:
        return []
