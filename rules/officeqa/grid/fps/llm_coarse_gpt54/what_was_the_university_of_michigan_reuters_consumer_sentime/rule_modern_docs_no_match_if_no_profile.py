def rule_modern_docs_no_match_if_no_profile(doc: dict) -> list[dict]:
    """Return nothing unless the document contains a Profile of the Economy section, useful for excluding older no-answer docs."""
    try:
        has_profile = any(
            "profile of the economy" in (((span.get("structure") or {}).get("path_text") or "").lower())
            or ("profile of the economy" in (span.get("text") or "").lower() and span.get("label") == "section_header")
            for span in doc.get("texts", [])
        )
        return [] if not has_profile else []
    except Exception:
        return []
