def rule_page1_cover_block_before_section12g(doc: dict) -> list[dict]:
    """Match spans between Section 12(b) and Section 12(g) on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        start = end = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "section 12(b)" in txt and start is None:
                start = i
            if span.get("page_no") == 1 and "section 12(g)" in txt and end is None:
                end = i
        if start is not None:
            if end is None:
                end = min(len(texts), start + 15)
            for j in range(start, min(end + 1, len(texts))):
                if texts[j].get("page_no") == 1:
                    out.append(texts[j])
        return out
    except Exception:
        return []
