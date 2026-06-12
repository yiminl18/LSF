def rule_profile_economy_deficit_and_publicly_held_debt(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention both deficit and publicly held debt."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit" in txt
                and ("debt held by the public" in txt or "publicly held debt" in txt)
            ):
                out.append(span)
    except Exception:
        return []
    return out
