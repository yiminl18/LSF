def rule_near_securities_registered_pursuant_12b(doc: dict) -> list[dict]:
    """Match spans near 'Securities registered pursuant to Section 12(b) of the Act'."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        anchor_idxs = []
        for i, span in enumerate(texts):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if re.search(r'securities registered pursuant to section 12\(b\) of the act', combined, re.I):
                anchor_idxs.append(i)
        for idx in anchor_idxs:
            for j in range(idx, min(len(texts), idx + 12)):
                s = texts[j]
                if s.get("page_no") == texts[idx].get("page_no"):
                    t = (s.get("text") or "").strip()
                    if re.search(r'new york stock exchange|nasdaq|global select market', t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
