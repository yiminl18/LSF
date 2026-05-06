def rule_total_obligations_tables(doc: dict) -> list[dict]:
    """Match tables containing 'total long-term obligations' keyword."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                "long-term obligations" in s.get("text", "").lower()]
    except Exception:
        return []
