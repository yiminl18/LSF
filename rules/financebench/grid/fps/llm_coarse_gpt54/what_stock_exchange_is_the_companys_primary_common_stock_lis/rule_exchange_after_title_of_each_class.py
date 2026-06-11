def rule_exchange_after_title_of_each_class(doc: dict) -> list[dict]:
    """Match exchange spans near the 'Title of each class' cover-page anchor."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            combined = " ".join([s.get("text", "") or "", s.get("text_span", "") or ""])
            if s.get("page_no") == 1 and re.search(r'title of each class', combined, re.I):
                for j in range(i, min(len(texts), i + 10)):
                    t = (texts[j].get("text") or "").strip()
                    if texts[j].get("page_no") == 1 and re.search(r'new york stock exchange|nasdaq|global select market', t, re.I):
                        out.append(texts[j])
        return out
    except Exception:
        return []
