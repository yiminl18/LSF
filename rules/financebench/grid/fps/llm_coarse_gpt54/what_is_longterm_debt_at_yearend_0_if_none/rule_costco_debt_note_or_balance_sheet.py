def rule_costco_debt_note_or_balance_sheet(doc: dict) -> list[dict]:
    """Match Costco debt-related tables or narrative spans mentioning long-term debt at year-end."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if span.get("label") == "table":
                if "balance sheet" in path or re.search(r"\blong[- ]term debt\b", txt, re.I):
                    out.append(span)
            else:
                if "costco" in path and re.search(r"\blong[- ]term debt\b", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
