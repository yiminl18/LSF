def rule_page1_text_with_both_state_and_ein_labels(doc: dict) -> list[dict]:
    """Match page-1 spans whose text contains both the incorporation label and the IRS label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if (
                span.get("page_no") == 1
                and re.search(r"State or other jurisdiction of incorporation|State or other jurisdiction of incorporation or organization", text, re.I)
                and re.search(r"I\.?R\.?S\.? Employer Identification", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
