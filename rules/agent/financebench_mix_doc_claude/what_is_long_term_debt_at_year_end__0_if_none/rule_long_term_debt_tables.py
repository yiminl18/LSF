def rule_long_term_debt_tables(doc: dict) -> list[dict]:
    """Match tables containing 'long-term debt' or 'long term debt' keyword."""
    try:
        return [s for s in doc.get("texts", [])
                if s.get("label") == "table" and
                ("long-term debt" in s.get("text", "").lower() or
                 "long term debt" in s.get("text", "").lower())]
    except Exception:
        return []
