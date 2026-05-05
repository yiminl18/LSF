def rule_phone_in_text_label(doc: dict) -> list[dict]:
    """Match text-label spans on page 1 containing a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "text":
                text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
                if phone_re.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
