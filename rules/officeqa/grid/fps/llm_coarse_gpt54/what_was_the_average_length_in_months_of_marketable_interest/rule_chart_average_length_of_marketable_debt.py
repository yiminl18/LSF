def rule_chart_average_length_of_marketable_debt(doc: dict) -> list[dict]:
    """Match chart/title spans about average length of the marketable debt, which often appear adjacent to the answer table."""
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "average length of the marketable debt" in txt:
                out.append(span)
    except Exception:
        return []
    return out
