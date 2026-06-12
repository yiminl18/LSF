def rule_profile_economy_federal_budget_deficit_exact_phrase(doc: dict) -> list[dict]:
    """Match spans containing the exact phrase 'federal budget deficit'."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("label") == "text" and "federal budget deficit" in txt.lower():
                out.append(span)
    except Exception:
        return []
    return out
