def rule_page1_short_bold_allcaps_after_common_stock(doc: dict) -> list[dict]:
    """Match short bold/all-caps spans after a common-stock mention, often the ticker."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"Common Stock", (s.get("text") or "") + " " + (s.get("text_span") or ""), re.I):
                for j in range(i + 1, min(len(texts), i + 6)):
                    t = (texts[j].get("text") or "").strip()
                    if texts[j].get("page_no") == 1 and len(t) <= 15 and re.fullmatch(r"[A-Z0-9./-]{1,15}", t):
                        out.append(texts[j])
        return out
    except Exception:
        return []
