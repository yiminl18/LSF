def rule_page1_cover_page_phone_pattern(doc: dict) -> list[dict]:
    """Match any page-1 cover-page span with a likely corporate phone number pattern."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(?:\+\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if phone_re.search(blob):
                out.append(span)
        return out
    except Exception:
        return []
