def rule_page12_section12b_listing_headers(doc: dict) -> list[dict]:
    """Match page-1/2 Section 12(b) header and label spans framing the listing block."""
    try:
        markers = (
            "securities registered pursuant to section 12(b)",
            "trading symbol",
            "trading symbol(s)",
            "title of each class",
            "name of each exchange on which registered",
            "name of exchange on which registered",
            "exchange on which registered",
        )
        return [
            s
            for s in doc.get("texts", [])
            if s.get("page_no", 999) <= 2
            and "indicate by check mark" not in s.get("text", "").lower()
            and any(marker in s.get("text", "").lower() for marker in markers)
        ]
    except Exception:
        return []
