def rule_page1_text_before_state_jurisdiction_label(doc: dict) -> list[dict]:
    """Match page-1 text/section_header spans immediately before a state/jurisdiction explanatory label."""
    try:
        texts = doc.get("texts", [])
        out = []
        markers = [
            "(State or other jurisdiction of incorporation)",
            "(State or other jurisdiction of incorporation or organization)",
            "State or other jurisdiction of incorporation",
            "State or other jurisdiction of incorporation or organization",
        ]
        for i in range(1, len(texts)):
            txt = texts[i].get("text") or ""
            if texts[i].get("page_no") == 1 and any(m in txt for m in markers):
                prev = texts[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
