def rule_spans_under_foreign_currency_positions(doc: dict) -> list[dict]:
    """Match spans under the FOREIGN CURRENCY POSITIONS section that mention Canadian dollar positions."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "foreign currency positions" in path and "canadian dollar positions" in txt:
                out.append(span)
        return out
    except Exception:
        return []
