def rule_federal_budget_and_debt_with_suspension(doc: dict) -> list[dict]:
    """Match suspension statements specifically within the Federal Budget and Debt subsection."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "federal budget and debt" in path.lower() and re.search(r"suspend|debt ceiling|debt limit", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
