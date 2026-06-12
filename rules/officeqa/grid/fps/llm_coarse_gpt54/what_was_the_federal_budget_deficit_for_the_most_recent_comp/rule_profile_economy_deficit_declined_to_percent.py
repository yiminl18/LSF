def rule_profile_economy_deficit_declined_to_percent(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans where the deficit declined/fell to a percent of GDP."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            low = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and re.search(r"deficit (declined|fell|narrowed).{0,80}\d+(\.\d+)?\s+percent of gdp", low)
            ):
                out.append(span)
    except Exception:
        return []
    return out
