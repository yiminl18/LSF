def rule_form_heading_near_mark_one(doc: dict) -> list[dict]:
    """Match form headings or report phrases near '(Mark One)' on the cover page."""
    import re
    try:
        texts = doc.get("texts", [])
        idxs = [i for i, s in enumerate(texts) if "(MARK ONE)" in ((s.get("text") or "").upper())]
        out = []
        for i in idxs:
            for j in range(max(0, i - 3), min(len(texts), i + 6)):
                s = texts[j]
                blob = ((s.get("text") or "") + " " + (s.get("text_span") or "")).upper()
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", blob, re.I) or "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob or "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob:
                    out.append(s)
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
