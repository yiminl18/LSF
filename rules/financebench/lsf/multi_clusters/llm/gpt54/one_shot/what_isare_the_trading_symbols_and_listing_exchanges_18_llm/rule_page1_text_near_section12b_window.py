def rule_page1_text_near_section12b_window(doc: dict) -> list[dict]:
    """Match page-1 text spans in a local window around Section 12(b) anchor."""
    try:
        texts = doc.get("texts", [])
        out = []
        idxs = [
            i for i, s in enumerate(texts)
            if s.get("page_no") == 1 and "section 12(b)" in (s.get("text") or "").lower()
        ]
        for i in idxs:
            for j in range(max(0, i - 3), min(len(texts), i + 10)):
                s = texts[j]
                if s.get("page_no") == 1 and s.get("label") in {"text", "section_header", "table"}:
                    out.append(s)
        return out
    except Exception:
        return []
