def rule_phone_number_only_or_labelled_on_page1(doc: dict) -> list[dict]:
    """Match page-1 spans that are either just a phone number or explicitly label it."""
    try:
        import re
        out = []
        only_re = re.compile(r"^\s*(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|[0-9]{3}[-/][0-9]{3}[-/][0-9]{4})\s*$")
        any_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|[0-9]{3}[-/][0-9]{3}[-/][0-9]{4})")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "") or ""
            if only_re.search(text) or (any_re.search(text) and re.search(r"telephone|area code", text, re.I)):
                out.append(span)
        return out
    except Exception:
        return []
