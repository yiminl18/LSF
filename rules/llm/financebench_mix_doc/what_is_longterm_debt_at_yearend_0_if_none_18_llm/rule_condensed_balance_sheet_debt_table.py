def rule_condensed_balance_sheet_debt_table(doc: dict) -> list[dict]:
    """Match condensed balance sheet tables with debt rows, common in 10-Qs."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "condensed consolidated balance" in text and re.search(r"\bdebt\b", text):
                out.append(span)
    except Exception:
        return []
    return out
