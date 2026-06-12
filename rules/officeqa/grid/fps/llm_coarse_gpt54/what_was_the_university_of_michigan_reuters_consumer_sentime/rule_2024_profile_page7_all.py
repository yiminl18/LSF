def rule_2024_profile_page7_all(doc: dict) -> list[dict]:
    """Match all page 7 Profile of the Economy spans as a narrow high-recall rule for recent template documents."""
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 7
            and "profile of the economy" in (((span.get("structure") or {}).get("path_text") or "").lower())
        ]
    except Exception:
        return []
