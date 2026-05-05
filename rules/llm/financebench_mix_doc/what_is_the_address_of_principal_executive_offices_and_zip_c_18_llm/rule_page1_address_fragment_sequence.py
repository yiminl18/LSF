def rule_page1_address_fragment_sequence(doc: dict) -> list[dict]:
    """Match consecutive page-1 spans that together form split address fragments."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 2):
            s1, s2, s3 = texts[i], texts[i+1], texts[i+2]
            if not all(s.get("page_no") == 1 for s in (s1, s2, s3)):
                continue
            t1, t2, t3 = (s1.get("text") or "").strip(), (s2.get("text") or "").strip(), (s3.get("text") or "").strip()
            combo = " ".join([t1, t2, t3]).strip()
            if re.search(r'^\d{1,5}\s', t1) and (re.search(r'\b\d{5}(?:-\d{4})?\b', combo) or re.search(r'BS30 8XP', combo, re.I)):
                out.extend([s1, s2, s3])
        return out
    except Exception:
        return []
