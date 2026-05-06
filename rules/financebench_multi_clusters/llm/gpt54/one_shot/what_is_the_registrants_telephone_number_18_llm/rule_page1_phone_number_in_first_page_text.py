def rule_page1_phone_number_in_first_page_text(doc: dict) -> list[dict]:
    """Match page-1 text spans that contain a phone number anywhere in text or text_span."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "text":
                blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
                if re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                    out.append(span)
        return out
    except Exception:
        return []
