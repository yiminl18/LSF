def rule_phone_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match phone-number spans near the '(Exact name of registrant...)' label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for i, span in enumerate(texts):
            if re.search(r"Exact name of registrant", span.get("text", "") or "", re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 12)):
                    cand = texts[j]
                    if cand.get("page_no") == span.get("page_no") and phone_re.search(cand.get("text", "") or ""):
                        out.append(cand)
        return out
    except Exception:
        return []
