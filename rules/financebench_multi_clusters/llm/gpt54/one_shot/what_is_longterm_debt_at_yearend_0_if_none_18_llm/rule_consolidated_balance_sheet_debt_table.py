def rule_consolidated_balance_sheet_debt_table(doc: dict) -> list[dict]:
    """Match consolidated balance sheet tables with debt rows, common in 10-Ks."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "consolidated balance sheet" in text and re.search(r"\bdebt\b", text):
                out.append(span)
    except Exception:
        return []
    return out
