def rule_page1_section12b_block(doc: dict) -> list[dict]:
    """Match spans on page 1 near 'Securities registered pursuant to Section 12(b) of the Act'."""
    try:
        texts = doc.get("texts", [])
        out = []
        trigger_idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if span.get("page_no") == 1 and "section 12(b)" in txt.lower():
                trigger_idxs.append(i)
        for i in trigger_idxs:
            for j in range(max(0, i - 3), min(len(texts), i + 12)):
                s = texts[j]
                if s.get("page_no") == 1:
                    out.append(s)
        return out
    except Exception:
        return []
