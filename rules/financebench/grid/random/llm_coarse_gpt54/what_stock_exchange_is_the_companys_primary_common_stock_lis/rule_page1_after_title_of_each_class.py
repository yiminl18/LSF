def rule_page1_after_title_of_each_class(doc: dict) -> list[dict]:
    """Return spans shortly after 'Title of each class' on page 1, where exchange value often appears in the same mini-block."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "title of each class" in txt:
                for j in range(i + 1, min(i + 8, len(texts))):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        break
                    out.append(s)
        return out
    except Exception:
        return []
