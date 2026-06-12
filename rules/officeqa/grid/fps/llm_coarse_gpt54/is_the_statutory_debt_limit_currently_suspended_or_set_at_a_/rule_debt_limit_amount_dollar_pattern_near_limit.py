def rule_debt_limit_amount_dollar_pattern_near_limit(doc: dict) -> list[dict]:
    """Match spans mentioning debt limit/ceiling together with a dollar-denominated amount."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt (limit|ceiling)|statutory (limit|limitation)", text, re.I) and re.search(r"\$[\d,]+(?:\.\d+)?\s*(million|billion|trillion)?", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
