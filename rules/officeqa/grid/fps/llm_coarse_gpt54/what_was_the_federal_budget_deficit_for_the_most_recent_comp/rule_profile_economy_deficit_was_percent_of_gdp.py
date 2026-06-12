def rule_profile_economy_deficit_was_percent_of_gdp(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans using the phrase 'deficit was X percent of GDP'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and re.search(r"deficit (was|fell|declined|rose).{0,80}\d+(\.\d+)?\s+percent of gdp", txt)
            ):
                out.append(span)
    except Exception:
        return []
    return out
