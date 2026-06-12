def rule_no_answer_pre_1990_no_profile(doc: dict) -> list[dict]:
    """Return empty when the document lacks a Profile of the Economy / GDP-growth style section."""
    try:
        has_profile = False
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "Profile of the Economy" in txt or "Profile of the Economy" in path:
                has_profile = True
                break
        return [] if not has_profile else []
    except Exception:
        return []
