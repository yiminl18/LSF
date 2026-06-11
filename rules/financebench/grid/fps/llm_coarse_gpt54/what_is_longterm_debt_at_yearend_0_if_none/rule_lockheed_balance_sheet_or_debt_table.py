def rule_lockheed_balance_sheet_or_debt_table(doc: dict) -> list[dict]:
    """Match Lockheed Martin balance sheet or debt tables likely containing long-term debt values."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            txt = (span.get("text") or "").lower()
            if "balance sheet" in path or "debt" in path:
                out.append(span)
            elif re.search(r"\blong[- ]term debt\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
