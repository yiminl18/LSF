def rule_phone_label_or_number_in_first_page_body(doc: dict) -> list[dict]:
    """Match first-page body/text spans that mention telephone or contain a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}-\d{3}-\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "list_item"}:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"telephone|area code|registrant", text, re.I) or phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
