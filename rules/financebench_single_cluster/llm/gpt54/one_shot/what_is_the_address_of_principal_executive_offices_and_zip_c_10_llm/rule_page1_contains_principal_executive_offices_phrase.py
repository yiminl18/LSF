def rule_page1_contains_principal_executive_offices_phrase(doc: dict) -> list[dict]:
    """Match any page-1 span containing the phrase principal executive offices."""
    try:
        spans = doc.get("texts", [])
        return [
            s for s in spans
            if s.get("page_no") == 1
            and "principal executive offices" in (((s.get("text") or "") + " " + (s.get("text_span") or "")).lower())
        ]
    except Exception:
        return []
