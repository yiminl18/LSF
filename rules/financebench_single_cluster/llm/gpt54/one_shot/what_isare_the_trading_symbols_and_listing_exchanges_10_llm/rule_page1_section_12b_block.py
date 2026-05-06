def rule_page1_section_12b_block(doc: dict) -> list[dict]:
    """Match spans on page 1 near 'Securities registered pursuant to Section 12(b)' and exchange/symbol labels."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"section\s+12\(b\)|12\(b\)\s+of\s+the\s+act", txt, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 12)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"trading symbol|symbol|exchange|registered|nasdaq|new york stock exchange|nyse", t, re.I):
                        out.append(s)
        return out
    except Exception:
        return []
