def rule_cover_page_before_securities_registered(doc: dict) -> list[dict]:
    """Match page-1 spans where phone text appears immediately before 'Securities registered'."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or ""))
            if span.get("page_no") == 1 and re.search(r"(telephone|area code).{0,120}securities registered pursuant to section 12\(b\)", text, re.I | re.S):
                out.append(span)
        return out
    except Exception:
        return []
