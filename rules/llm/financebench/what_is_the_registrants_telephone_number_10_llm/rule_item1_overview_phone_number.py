def rule_item1_overview_phone_number(doc: dict) -> list[dict]:
    """Match Item 1/Overview spans containing a phone-number-like pattern."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "").lower()
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if ("item 1" in path or "overview" in path or "item 1" in text.lower()) and phone_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
