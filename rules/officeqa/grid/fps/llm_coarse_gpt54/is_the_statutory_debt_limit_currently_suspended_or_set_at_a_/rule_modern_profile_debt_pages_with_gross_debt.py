def rule_modern_profile_debt_pages_with_gross_debt(doc: dict) -> list[dict]:
    """Match modern Federal Budget and Debt spans mentioning gross federal debt or debt held by the public."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "federal budget and debt" in path.lower() and re.search(r"gross federal debt|debt held by the public|debt ceiling|debt limit|suspended", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
