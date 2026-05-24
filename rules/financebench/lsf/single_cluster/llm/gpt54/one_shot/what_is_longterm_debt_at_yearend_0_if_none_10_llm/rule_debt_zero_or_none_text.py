def rule_debt_zero_or_none_text(doc: dict) -> list[dict]:
    """Match spans indicating no long-term debt or zero debt."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term debt\b", txt) and (
                "none" in txt or re.search(r"\b0\b", txt) or "no long-term debt" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
