def rule_near_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Match spans with phone numbers near the 'Address of principal executive offices' label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if re.search(r"Address of principal executive offices", text, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    cand = texts[j]
                    if cand.get("page_no") == span.get("page_no") and phone_re.search(cand.get("text", "") or ""):
                        out.append(cand)
        return out
    except Exception:
        return []
