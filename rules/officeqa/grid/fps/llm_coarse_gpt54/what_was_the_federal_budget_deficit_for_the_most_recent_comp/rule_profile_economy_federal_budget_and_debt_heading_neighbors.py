def rule_profile_economy_federal_budget_and_debt_heading_neighbors(doc: dict) -> list[dict]:
    """Match the Federal Budget and Debt heading and nearby explanatory text."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("label") == "section_header" and "federal budget and debt" in txt:
                out.append(span)
                for j in range(i + 1, min(i + 6, len(texts))):
                    s2 = texts[j]
                    if s2.get("label") == "section_header":
                        break
                    if "deficit" in (s2.get("text") or "").lower():
                        out.append(s2)
    except Exception:
        return []
    return out
