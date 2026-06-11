def rule_american_water_balance_sheet_long_term_debt_table(doc: dict) -> list[dict]:
    """Match American Water-style consolidated balance sheet tables with long-term debt rows."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            text = (span.get("text") or "").lower()
            if "american water works" in path and "balance" in path:
                out.append(span)
                continue
            if re.search(r"\blong[- ]term debt\b", text) and re.search(r"\bcurrent liabilities\b|\btotal liabilities\b", text):
                out.append(span)
        return out
    except Exception:
        return []
