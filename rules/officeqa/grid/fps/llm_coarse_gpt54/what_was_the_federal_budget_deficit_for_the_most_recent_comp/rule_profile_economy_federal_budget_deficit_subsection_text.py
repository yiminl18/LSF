def rule_profile_economy_federal_budget_deficit_subsection_text(doc: dict) -> list[dict]:
    """Match text spans whose path_text includes the Federal budget deficit subsection."""
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") == "text" and "federal budget deficit" in path:
                out.append(span)
    except Exception:
        return []
    return out
