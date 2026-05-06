def rule_page1_bold_text_before_state_caption(doc: dict) -> list[dict]:
    """Match bold page-1 text spans immediately before the state-of-incorporation caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            cur = texts[i]
            prev = texts[i - 1]
            if (
                cur.get("page_no") == 1
                and "state or other jurisdiction of incorporation" in (cur.get("text", "") or "").lower()
                and prev.get("page_no") == 1
                and prev.get("bold") == 1
            ):
                out.append(prev)
        return out
    except Exception:
        return []
