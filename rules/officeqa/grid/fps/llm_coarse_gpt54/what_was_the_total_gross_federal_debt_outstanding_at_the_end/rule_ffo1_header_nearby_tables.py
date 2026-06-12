def rule_ffo1_header_nearby_tables(doc: dict) -> list[dict]:
    """Match tables immediately following a section header mentioning Table FFO-1 or Summary of Fiscal Operations."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = span.get("text", "") or ""
            if not (re.search(r'ffo[\-–— ]?1', txt, re.I) or re.search(r'summary of fiscal operations', txt, re.I)):
                continue
            for j in range(i + 1, min(i + 6, len(texts))):
                nxt = texts[j]
                if nxt.get("label") == "table":
                    out.append(nxt)
    except Exception:
        return []
    return out
