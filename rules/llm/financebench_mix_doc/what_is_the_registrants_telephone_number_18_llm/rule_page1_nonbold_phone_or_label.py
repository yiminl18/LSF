def rule_page1_nonbold_phone_or_label(doc: dict) -> list[dict]:
    """Match non-bold page-1 spans that are either the phone number or its label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("bold") != 0:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$", (span.get("text", "") or "").strip()):
                out.append(span)
        return out
    except Exception:
        return []
