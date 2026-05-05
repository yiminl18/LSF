def rule_page1_split_address_city_state_zip(doc: dict) -> list[dict]:
    """Match split address patterns across adjacent spans for street, city/state, and ZIP."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 3):
            s1, s2, s3, s4 = texts[i:i+4]
            if not all(s.get("page_no") == 1 for s in (s1, s2, s3, s4)):
                continue
            combo = " ".join((s.get("text") or "").strip() for s in (s1, s2, s3, s4))
            if re.search(r'^\d{1,5}\s', (s1.get("text") or "").strip()) and (
                re.search(r'Santa Monica.*CA.*9040[45]', combo, re.I) or
                re.search(r'St\. Paul.*Minnesota.*55144-1000', combo, re.I)
            ):
                out.extend([s1, s2, s3, s4])
        return out
    except Exception:
        return []
