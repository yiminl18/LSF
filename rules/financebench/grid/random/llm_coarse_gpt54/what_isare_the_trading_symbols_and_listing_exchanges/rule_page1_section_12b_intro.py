def rule_page1_section_12b_intro(doc: dict) -> list[dict]:
    """Match page-1 spans introducing the Section 12(b) securities registration block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(b\) of the act", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
