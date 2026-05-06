def rule_page1_phone_pattern(doc: dict) -> list[dict]:
    """Match page 1 spans containing phone number patterns."""
    try:
        import re
        results = []
        phone_patterns = [
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{3}-\d{3}-\d{4}',
            r'\+\d{2}\s+\d{3}\s+\d+',
        ]
        combined_pattern = '|'.join(phone_patterns)
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "")
            if re.search(combined_pattern, text):
                results.append(span)
        return results
    except Exception:
        return []
