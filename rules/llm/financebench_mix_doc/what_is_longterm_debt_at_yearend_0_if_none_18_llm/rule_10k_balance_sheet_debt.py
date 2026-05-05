def rule_10k_balance_sheet_debt(doc: dict) -> list[dict]:
    """Match 10-K balance sheet tables with debt values."""
    out = []
    try:
        is_10k = any("form 10-k" in (s.get("text") or "").lower() for s in doc.get("texts", []))
        if not is_10k:
            return []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if ("balance sheet" in txt or "balance sheets" in txt) and "debt" in txt:
                    out.append(span)
    except Exception:
        return []
    return out
