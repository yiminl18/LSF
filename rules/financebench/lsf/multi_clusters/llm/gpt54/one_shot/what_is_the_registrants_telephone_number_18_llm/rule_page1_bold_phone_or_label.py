def rule_page1_bold_phone_or_label(doc: dict) -> list[dict]:
    """Match bold page-1 spans that are either the phone number or its label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$", (span.get("text", "") or "").strip()):
                out.append(span)
        return out
    except Exception:
        return []
