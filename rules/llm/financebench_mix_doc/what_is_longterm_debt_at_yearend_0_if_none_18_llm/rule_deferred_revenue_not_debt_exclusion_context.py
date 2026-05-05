def rule_deferred_revenue_not_debt_exclusion_context(doc: dict) -> list[dict]:
    """Match debt-related spans while excluding pure deferred revenue-only contexts."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if re.search(r"\bdebt\b|\blong[- ]term debt\b|\bborrowings\b", txt):
                if "deferred revenue" in txt and not re.search(r"\bdebt\b|\blong[- ]term debt\b", txt):
                    continue
                out.append(span)
    except Exception:
        return []
    return out
