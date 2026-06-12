def rule_tables_with_weekly_bank_positions_and_canadian(doc: dict) -> list[dict]:
    """Match tables mentioning both Canadian dollar positions and weekly bank positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "canadian" in txt and "weekly bank positions" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
