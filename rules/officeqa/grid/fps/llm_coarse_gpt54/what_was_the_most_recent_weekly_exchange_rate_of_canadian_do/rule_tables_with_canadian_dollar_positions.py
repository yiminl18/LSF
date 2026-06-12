def rule_tables_with_canadian_dollar_positions(doc: dict) -> list[dict]:
    """Match table spans whose text mentions Canadian dollar positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "canadian dollar positions" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
