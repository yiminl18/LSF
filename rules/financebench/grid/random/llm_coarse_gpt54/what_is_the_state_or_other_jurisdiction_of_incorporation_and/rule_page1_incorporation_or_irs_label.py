def rule_page1_incorporation_or_irs_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing the incorporation or IRS label text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"State or other jurisdiction of incorporation|I\.?R\.?S\.? Employer Identification", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
