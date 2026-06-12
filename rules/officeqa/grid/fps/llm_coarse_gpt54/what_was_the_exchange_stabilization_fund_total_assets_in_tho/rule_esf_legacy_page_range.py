def rule_esf_legacy_page_range(doc: dict) -> list[dict]:
    """Match tables in pages 70-170, common for ESF in 1980s-1990s bulletins."""
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table"
            and isinstance(s.get("page_no"), int)
            and 70 <= s.get("page_no") <= 170
        ]
    except Exception:
        return []
