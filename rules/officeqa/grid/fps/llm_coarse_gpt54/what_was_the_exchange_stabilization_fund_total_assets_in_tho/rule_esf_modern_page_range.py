def rule_esf_modern_page_range(doc: dict) -> list[dict]:
    """Match tables in pages 60-110, common for ESF in modern bulletins."""
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table"
            and isinstance(s.get("page_no"), int)
            and 60 <= s.get("page_no") <= 110
        ]
    except Exception:
        return []
