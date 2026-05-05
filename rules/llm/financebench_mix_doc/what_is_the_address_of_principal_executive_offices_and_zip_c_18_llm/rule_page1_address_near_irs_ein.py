def rule_page1_address_near_irs_ein(doc: dict) -> list[dict]:
    """Match spans near the IRS Employer Identification label that look like address content."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        idxs = [i for i, s in enumerate(texts) if s.get("page_no") == 1 and re.search(r'i\.r\.s\. employer identification|employer identification no', (s.get("text") or "") + " " + (s.get("text_span") or ""), re.I)]
        for idx in idxs:
            for j in range(max(0, idx - 3), min(len(texts), idx + 8)):
                s = texts[j]
                if s.get("page_no") != 1:
                    continue
                t = (s.get("text") or "").strip()
                if re.search(r'^\d{1,5}\s', t) or re.search(r'^\d{5}(?:-\d{4})?$', t) or re.search(r'United Kingdom|Bristol|California|Washington|Minnesota', t, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
