def rule_profile_economy_federal_budget_and_debt_pages(doc: dict) -> list[dict]:
    """Match pages/spans in Profile of the Economy where the Federal Budget and Debt subsection appears."""
    out = []
    try:
        texts = doc.get("texts", [])
        active = False
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("label") == "section_header":
                if "federal budget and debt" in txt or "federal budget deficit" in txt:
                    active = True
                    out.append(span)
                    continue
                if active:
                    active = False
            if active and span.get("label") == "text":
                out.append(span)
    except Exception:
        return []
    return out
