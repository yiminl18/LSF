def rule_page1_near_12b_block(doc: dict) -> list[dict]:
    """Return spans within a short window after the Section 12(b) intro on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(b\) of the act", txt, re.I):
                for j in range(i, min(i + 12, len(texts))):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
