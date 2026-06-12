def rule_federal_budget_and_debt_section(doc: dict) -> list[dict]:
    """Match spans under the modern Profile of the Economy subsection Federal Budget and Debt."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "federal budget and debt" in path.lower() or text.strip().lower() == "federal budget and debt":
                out.append(span)
        return out
    except Exception:
        return []
