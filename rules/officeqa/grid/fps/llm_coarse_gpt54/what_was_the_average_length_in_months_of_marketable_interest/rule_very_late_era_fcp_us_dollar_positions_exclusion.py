def rule_very_late_era_fcp_us_dollar_positions_exclusion(doc: dict) -> list[dict]:
    """High-recall rule: match average-length spans while excluding foreign currency position pages that also use month counts."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "average length" in txt and "foreign currency positions" not in path:
                out.append(span)
    except Exception:
        return []
    return out
