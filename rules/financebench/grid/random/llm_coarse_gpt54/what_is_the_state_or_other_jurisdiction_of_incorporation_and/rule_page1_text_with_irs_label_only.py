def rule_page1_text_with_irs_label_only(doc: dict) -> list[dict]:
    """Match page-1 spans containing only the IRS Employer Identification label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"I\.?R\.?S\.? Employer Identification No\.?|IRS Employer Identification No\.?", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
