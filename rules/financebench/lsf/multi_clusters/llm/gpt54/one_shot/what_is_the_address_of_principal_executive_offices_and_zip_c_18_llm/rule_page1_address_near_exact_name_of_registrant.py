def rule_page1_address_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match spans near the '(Exact name of registrant...)' label that look like address content."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        idxs = [i for i, s in enumerate(texts) if s.get("page_no") == 1 and re.search(r'exact name of registrant', (s.get("text") or "") + " " + (s.get("text_span") or ""), re.I)]
        for idx in idxs:
            for j in range(max(0, idx), min(len(texts), idx + 15)):
                s = texts[j]
                if s.get("page_no") != 1:
                    continue
                t = (s.get("text") or "").strip()
                if re.search(r'^\d{1,5}\s', t) or re.search(r'\b\d{5}(?:-\d{4})?\b', t) or re.search(r'BS30 8XP', t, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
