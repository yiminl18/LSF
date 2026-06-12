def rule_profile_economy_deficit_numeric_percent_pattern(doc: dict) -> list[dict]:
    """Match Profile of the Economy text with numeric percent values tied to deficit."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "profile of the economy" not in path:
                continue
            if "deficit" in low and re.search(r"\d+(\.\d+)?\s+percent of gdp", low):
                out.append(span)
    except Exception:
        return []
    return out
