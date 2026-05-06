def rule_debt_zero_for_8k_forms(doc: dict) -> list[dict]:
    """Match 8-K documents where the answer should often be 0 because no financial debt table exists."""
    out = []
    try:
        has_8k = False
        has_debt_table = False
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "form 8-k" in txt:
                has_8k = True
            if span.get("label") == "table" and ("balance sheet" in txt or "long-term debt" in txt or "debt" in txt):
                has_debt_table = True
        if has_8k and not has_debt_table:
            for span in doc.get("texts", []):
                if "form 8-k" in (span.get("text") or "").lower():
                    out.append(span)
    except Exception:
        return []
    return out
