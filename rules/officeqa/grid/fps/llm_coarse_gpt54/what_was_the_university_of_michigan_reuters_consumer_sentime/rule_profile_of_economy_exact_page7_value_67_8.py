def rule_profile_of_economy_exact_page7_value_67_8(doc: dict) -> list[dict]:
    """Match page-7 Profile of the Economy spans containing the exact value 67.8."""
    import re
    try:
        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 7
            and "profile of the economy" in (((span.get("structure") or {}).get("path_text") or "").lower())
            and re.search(r"\b67\.8\b", (span.get("text") or ""))
        ]
    except Exception:
        return []
