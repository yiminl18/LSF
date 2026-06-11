def rule_page1_address_of_principal_executive_offices_text_only(doc: dict) -> list[dict]:
    """Match text-only spans on page 1 that are exactly the address label."""
    try:
        import re
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") == "text"
            and re.fullmatch(r'\(?Address of principal executive offices\)?(?: \(Zip Code\))?', (span.get("text") or "").strip(), re.I)
        ]
    except Exception:
        return []
