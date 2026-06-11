def rule_condensed_consolidated_balance_sheet_debt(doc: dict) -> list[dict]:
    """Match condensed consolidated balance sheet tables with debt rows, common in 10-Qs."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            combo = ((span.get("structure") or {}).get("path_text", "") or "") + " " + (span.get("text", "") or "")
            if re.search(r"condensed consolidated balance sheet", combo, re.I) and re.search(r"\bdebt\b", combo, re.I):
                out.append(span)
        return out
    except Exception:
        return []
