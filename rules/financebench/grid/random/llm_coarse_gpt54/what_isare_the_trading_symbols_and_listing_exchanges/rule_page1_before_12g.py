def rule_page1_before_12g(doc: dict) -> list[dict]:
    """Return page-1 spans immediately before the 'Securities registered pursuant to Section 12(g)' line."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(g\) of the act", txt, re.I):
                for j in range(max(0, i - 6), i):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
