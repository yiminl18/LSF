def rule_profile_economy_deficit_fell_from_peak(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans saying the deficit fell from a peak to a later fiscal year percent of GDP."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit has fallen" in low or "deficit fell" in low
            ):
                if "percent of gdp" in low:
                    out.append(span)
    except Exception:
        return []
    return out
