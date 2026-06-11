def rule_page1_after_section12g_stop(doc: dict) -> list[dict]:
    """Return exchange-like spans on page 1 before the Section 12(g) block begins."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and start is None and "securities registered pursuant to section 12(b)" in txt:
                start = i
            if span.get("page_no") == 1 and "securities registered pursuant to section 12(g)" in txt:
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 20)
        pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
        ]
        for j in range(start, end):
            s = texts[j]
            if s.get("page_no") != 1:
                continue
            low = (s.get("text") or "").lower()
            if any(re.search(p, low) for p in pats):
                out.append(s)
        return out
    except Exception:
        return []
