def rule_cover_page_section12b_same_span(doc: dict) -> list[dict]:
    """Match page-1 spans whose text_span embeds the entire Section 12(b) securities block."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            blob = (s.get("text") or "") + " " + (s.get("text_span") or "")
            if re.search(r"Section 12\(b\)", blob, re.I) and re.search(r"Trading Symbol|Trading symbol|exchange on which registered", blob, re.I):
                out.append(s)
        return out
    except Exception:
        return []
