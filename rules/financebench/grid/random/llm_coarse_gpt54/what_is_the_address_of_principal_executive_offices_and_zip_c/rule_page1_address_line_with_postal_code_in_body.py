def rule_page1_address_line_with_postal_code_in_body(doc: dict) -> list[dict]:
    """Match page-1 body text containing a full address and postal code."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and ((span.get("structure") or {}).get("level") or "") == "Body"
            and re.search(r'\d{2,}.*(?:\d{5}(?:-\d{4})?|BS30 ?8XP)', (span.get("text") or ""), re.I)
        ]
    except Exception:
        return []
