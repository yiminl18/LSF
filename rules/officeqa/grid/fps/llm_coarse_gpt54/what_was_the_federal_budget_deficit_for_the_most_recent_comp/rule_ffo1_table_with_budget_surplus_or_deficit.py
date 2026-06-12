def rule_ffo1_table_with_budget_surplus_or_deficit(doc: dict) -> list[dict]:
    """Match FFO-1 tables containing 'budget surplus or deficit' wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "summary of fiscal operations" in txt and "budget surplus or deficit" in txt:
                    out.append(span)
    except Exception:
        return []
    return out
