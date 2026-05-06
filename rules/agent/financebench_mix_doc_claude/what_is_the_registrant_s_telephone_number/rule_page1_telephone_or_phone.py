def rule_page1_telephone_or_phone(doc: dict) -> list[dict]:
    """Return page 1 spans containing telephone keyword or phone number pattern."""
    try:
        import re
        results = []
        texts = doc.get("texts", [])
        phone_pattern = re.compile(r'[\+\(]?\d{2,3}[\)\s\-]?\s*\d{3}[\s\-]?\d{4}')

        for span in texts:
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            text_lower = text.lower()
            if "telephone" in text_lower:
                results.append(span)
            elif phone_pattern.search(text):
                results.append(span)

        return results
    except Exception:
        return []
