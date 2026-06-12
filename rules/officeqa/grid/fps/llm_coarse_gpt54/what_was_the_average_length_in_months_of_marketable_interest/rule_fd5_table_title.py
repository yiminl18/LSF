def rule_fd5_table_title(doc: dict) -> list[dict]:
    """Match table/title spans for the Federal Debt table about maturity distribution and average length of marketable debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if (
                "maturity distribution" in txt
                and "average length" in txt
                and "marketable" in txt
                and "debt" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
