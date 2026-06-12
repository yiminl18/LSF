def rule_international_statistics_contents_with_foreign_currency_positions(doc: dict) -> list[dict]:
    """Match contents spans where foreign currency positions and Canadian dollar positions are listed together."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "foreign currency positions" in txt and "canadian dollar positions" in txt:
                out.append(span)
        return out
    except Exception:
        return []
