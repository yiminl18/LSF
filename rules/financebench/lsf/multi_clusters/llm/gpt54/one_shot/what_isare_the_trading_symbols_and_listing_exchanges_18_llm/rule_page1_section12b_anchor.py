def rule_page1_section12b_anchor(doc: dict) -> list[dict]:
    """Match spans on page 1 near 'Securities registered pursuant to Section 12(b) of the Act'."""
    try:
        texts = doc.get("texts", [])
        out = []
        anchor_idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "securities registered pursuant to section 12(b)" in txt:
                anchor_idxs.append(i)
        for i in anchor_idxs:
            for j in range(max(0, i - 2), min(len(texts), i + 12)):
                s = texts[j]
                if s.get("page_no") == 1:
                    out.append(s)
        return out
    except Exception:
        return []
