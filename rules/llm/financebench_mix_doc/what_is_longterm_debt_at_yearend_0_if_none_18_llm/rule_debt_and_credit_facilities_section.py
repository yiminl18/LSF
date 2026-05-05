def rule_debt_and_credit_facilities_section(doc: dict) -> list[dict]:
    """Match sections or spans mentioning debt and credit facilities."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if re.search(r"debt and credit facilities|credit facilities|long-term debt", txt) or re.search(r"debt and credit facilities|credit facilities|long-term debt", path):
                out.append(span)
    except Exception:
        return []
    return out
