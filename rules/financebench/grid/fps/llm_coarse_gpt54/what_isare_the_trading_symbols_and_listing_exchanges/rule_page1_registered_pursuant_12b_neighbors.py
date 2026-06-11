def rule_page1_registered_pursuant_12b_neighbors(doc: dict) -> list[dict]:
    """Match neighbors of page-1 spans mentioning securities registered pursuant to Section 12(b)."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            txt = (s.get("text") or "")
            if s.get("page_no") == 1 and re.search(r"Securities registered pursuant to Section 12\(b\)", txt, re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 12)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
