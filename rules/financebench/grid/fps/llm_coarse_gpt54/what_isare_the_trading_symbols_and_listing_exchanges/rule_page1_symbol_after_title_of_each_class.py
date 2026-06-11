def rule_page1_symbol_after_title_of_each_class(doc: dict) -> list[dict]:
    """Match ticker-like spans shortly after 'Title of each class' on page 1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, s in enumerate(texts):
            if s.get("page_no") == 1 and re.search(r"Title of each class", (s.get("text") or ""), re.I):
                for j in range(i + 1, min(len(texts), i + 8)):
                    t = (texts[j].get("text") or "").strip()
                    if texts[j].get("page_no") == 1 and re.fullmatch(r"[A-Z0-9./-]{1,15}", t):
                        out.append(texts[j])
        return out
    except Exception:
        return []
