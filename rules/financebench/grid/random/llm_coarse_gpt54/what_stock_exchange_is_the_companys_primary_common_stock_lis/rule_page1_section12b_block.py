def rule_page1_section12b_block(doc: dict) -> list[dict]:
    """Match spans on page 1 near the 'Securities registered pursuant to Section 12(b) of the Act' block."""
    try:
        texts = doc.get("texts", [])
        out = []
        anchor_idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "securities registered pursuant to section 12(b)" in txt:
                anchor_idxs.append(i)
        for i in anchor_idxs:
            for j in range(i, min(i + 12, len(texts))):
                s = texts[j]
                if s.get("page_no") != 1:
                    break
                out.append(s)
        return out
    except Exception:
        return []
