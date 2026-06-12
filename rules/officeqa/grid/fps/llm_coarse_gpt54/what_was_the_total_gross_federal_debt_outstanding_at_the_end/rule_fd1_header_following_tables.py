def rule_fd1_header_following_tables(doc: dict) -> list[dict]:
    """Match tables immediately after a section header for FD-1 / Summary of Federal Debt."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            txt = span.get("text", "") or ""
            if not (re.search(r'fd[\-–— ]?1', txt, re.I) or re.search(r'summary of federal debt', txt, re.I)):
                continue
            for j in range(i + 1, min(i + 6, len(texts))):
                nxt = texts[j]
                if nxt.get("label") == "table":
                    out.append(nxt)
    except Exception:
        return []
    return out
