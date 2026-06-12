def rule_tables_under_contents_foreign_currency_positions(doc: dict) -> list[dict]:
    """Match contents tables under FOREIGN CURRENCY POSITIONS that list Canadian dollar positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "foreign currency positions" in path and "canadian dollar positions" in txt:
                out.append(span)
        return out
    except Exception:
        return []
