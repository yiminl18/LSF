def rule_form_heading_with_commission_file_nearby(doc: dict) -> list[dict]:
    """Match form/report spans that occur near 'Commission File Number/No.' on the cover page."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            blob = ((s.get("text") or "") + " " + (s.get("text_span") or "")).upper()
            if "COMMISSION FILE NUMBER" in blob or "COMMISSION FILE NO." in blob or "COMMISSION FILE NO" in blob:
                for j in range(max(0, i - 6), min(len(texts), i + 3)):
                    t = ((texts[j].get("text") or "") + " " + (texts[j].get("text_span") or "")).upper()
                    if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I) or "CURRENT REPORT" in t:
                        out.append(texts[j])
        seen = set()
        dedup = []
        for s in out:
            key = id(s)
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup
    except Exception:
        return []
