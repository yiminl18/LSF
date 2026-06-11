def rule_form_prefix_page1(doc: dict) -> list[dict]:
    """Match page-1 spans beginning with FORM followed by a form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.match(r"^FORM\s+[A-Z0-9\-]+", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
