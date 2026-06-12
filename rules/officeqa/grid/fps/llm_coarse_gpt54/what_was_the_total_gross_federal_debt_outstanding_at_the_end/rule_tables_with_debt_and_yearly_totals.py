def rule_tables_with_debt_and_yearly_totals(doc: dict) -> list[dict]:
    """Match debt-related tables with many yearly totals, useful for historical summary tables."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text", "") or ""
            if not re.search(r'debt', txt, re.I):
                continue
            years = re.findall(r'\b(19[6-9]\d|20[0-2]\d)\b', txt)
            nums = re.findall(r'\b\d{6,}\b', txt.replace(",", ""))
            if len(set(years)) >= 3 and len(nums) >= 3:
                out.append(span)
    except Exception:
        return []
    return out
