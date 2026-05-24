def rule_long_term_debt_near_total_liabilities(doc: dict) -> list[dict]:
    """Match tables where long-term debt appears near total liabilities."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if re.search(r"long[- ]term debt", txt) and "total liabilities" in txt:
                out.append(span)
    except Exception:
        return []
    return out
