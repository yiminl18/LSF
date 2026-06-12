def rule_profile_economy_budget_deficit_sentence_with_peak(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that compare current deficit to a historical peak."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            low = txt.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "peak" in low
                and "fiscal year" in low
                and "deficit" in low
                and "gdp" in low
            ):
                out.append(span)
    except Exception:
        return []
    return out
