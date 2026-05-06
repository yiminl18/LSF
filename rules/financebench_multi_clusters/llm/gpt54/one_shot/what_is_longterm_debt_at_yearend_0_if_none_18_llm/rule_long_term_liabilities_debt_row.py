def rule_long_term_liabilities_debt_row(doc: dict) -> list[dict]:
    """Match tables where long-term liabilities section includes debt."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            if "long-term liabilities" in text and re.search(r"\bdebt\b", text):
                out.append(span)
    except Exception:
        return []
    return out
