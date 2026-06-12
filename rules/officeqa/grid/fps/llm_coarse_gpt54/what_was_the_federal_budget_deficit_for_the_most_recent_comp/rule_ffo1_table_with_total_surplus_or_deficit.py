def rule_ffo1_table_with_total_surplus_or_deficit(doc: dict) -> list[dict]:
    """Match FFO-1 tables containing 'total surplus or deficit' wording."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "summary of fiscal operations" in txt and "total surplus or deficit" in txt:
                    out.append(span)
    except Exception:
        return []
    return out
