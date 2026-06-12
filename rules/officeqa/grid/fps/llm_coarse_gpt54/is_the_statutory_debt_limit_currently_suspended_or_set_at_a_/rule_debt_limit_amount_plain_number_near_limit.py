def rule_debt_limit_amount_plain_number_near_limit(doc: dict) -> list[dict]:
    """Match spans mentioning debt limit/ceiling together with a large plain-number amount and unit."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if re.search(r"debt (limit|ceiling)|statutory (limit|limitation)", text, re.I) and re.search(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(million|billion|trillion)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
