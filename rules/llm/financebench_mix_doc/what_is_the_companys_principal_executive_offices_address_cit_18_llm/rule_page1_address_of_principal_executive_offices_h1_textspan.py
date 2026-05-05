def rule_page1_address_of_principal_executive_offices_h1_textspan(doc: dict) -> list[dict]:
    """Match H1 company spans on page 1 whose text_span includes the principal executive offices phrase."""
    try:
        texts = doc.get("texts", [])
        return [
            span for span in texts
            if span.get("page_no") == 1
            and span.get("label") == "section_header"
            and span.get("structure", {}).get("level") == "H1"
            and "principal executive offices" in (span.get("text_span") or "").lower()
        ]
    except Exception:
        return []
