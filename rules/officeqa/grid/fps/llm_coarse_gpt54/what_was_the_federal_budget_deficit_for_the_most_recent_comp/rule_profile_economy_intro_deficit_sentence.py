def rule_profile_economy_intro_deficit_sentence(doc: dict) -> list[dict]:
    """Match introductory Profile of the Economy paragraphs that state the latest completed fiscal-year deficit percent."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "federal budget deficit" in low
                and ("fiscal year 20" in low or "fiscal year 19" in low)
                and "percent of gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
