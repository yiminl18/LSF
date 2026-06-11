def rule_10q_financial_table_period_headers(doc: dict) -> list[dict]:
    """Match tables whose column headers say 'For the Three Months Ended' or 'For the Six Months Ended' with a date."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            joined = " ".join((c.get("text") or "").lower() for c in cells)
            if "for the three months ended" in joined or "for the six months ended" in joined:
                out.append(span)
        return out
    except Exception:
        return []
