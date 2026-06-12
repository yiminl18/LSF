def rule_profile_economy_deficit_of_gdp_text(doc: dict) -> list[dict]:
    """Match Profile of the Economy text spans containing 'deficit' and 'of GDP'."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = txt.lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path.lower()
                and "deficit" in low
                and "of gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
