def rule_q10q_condensed_balance_sheet_debt(doc: dict) -> list[dict]:
    """Match 10-Q condensed balance sheet tables with debt values."""
    out = []
    try:
        is_10q = any("form 10-q" in (s.get("text") or "").lower() for s in doc.get("texts", []))
        if not is_10q:
            return []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "condensed consolidated balance" in txt and "debt" in txt:
                    out.append(span)
    except Exception:
        return []
    return out
