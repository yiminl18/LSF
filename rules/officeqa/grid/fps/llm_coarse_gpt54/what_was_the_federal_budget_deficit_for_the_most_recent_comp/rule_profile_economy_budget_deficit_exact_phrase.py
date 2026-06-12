def rule_profile_economy_budget_deficit_exact_phrase(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'budget deficit' in the Profile of the Economy article."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if span.get("label") == "text" and "profile of the economy" in path.lower() and "budget deficit" in txt.lower():
                out.append(span)
    except Exception:
        return []
    return out
