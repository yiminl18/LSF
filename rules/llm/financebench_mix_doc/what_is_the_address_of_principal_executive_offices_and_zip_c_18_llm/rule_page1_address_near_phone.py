def rule_page1_address_near_phone(doc: dict) -> list[dict]:
    """Match spans near a phone-number span that look like address content."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        phone_idxs = [i for i, s in enumerate(texts) if s.get("page_no") == 1 and re.search(r'\(?\+?\d[\d\-\)\( ]{6,}', (s.get("text") or ""))]
        for idx in phone_idxs:
            for j in range(max(0, idx - 6), min(len(texts), idx + 2)):
                s = texts[j]
                if s.get("page_no") != 1:
                    continue
                t = (s.get("text") or "").strip()
                if re.search(r'^\d{1,5}\s', t) or re.search(r'^\d{5}(?:-\d{4})?$', t) or re.search(r'United Kingdom|Bristol|California|Washington|Minnesota|Santa Monica,', t, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
