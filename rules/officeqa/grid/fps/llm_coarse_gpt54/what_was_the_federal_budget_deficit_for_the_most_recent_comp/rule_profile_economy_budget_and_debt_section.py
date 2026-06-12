def rule_profile_economy_budget_and_debt_section(doc: dict) -> list[dict]:
    """Match spans under the 'Federal Budget and Debt' subsection in Profile of the Economy."""
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("label") == "section_header" and "federal budget and debt" in txt.lower():
                out.append(span)
                for j in range(i + 1, min(i + 8, len(texts))):
                    s2 = texts[j]
                    t2 = (s2.get("text") or "").lower()
                    if s2.get("label") == "section_header":
                        break
                    if "deficit" in t2 and "gdp" in t2:
                        out.append(s2)
            elif "profile of the economy" in path and "federal budget and debt" in path:
                if "deficit" in txt.lower() and "gdp" in txt.lower():
                    out.append(span)
    except Exception:
        return []
    return out
