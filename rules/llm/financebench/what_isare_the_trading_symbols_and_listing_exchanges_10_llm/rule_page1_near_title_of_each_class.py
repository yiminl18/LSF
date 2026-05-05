def rule_page1_near_title_of_each_class(doc: dict) -> list[dict]:
    """Match spans around 'Title of each class' because symbol/exchange values usually follow immediately."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"title of each class", txt, re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 8)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"title of each class|trading symbol|exchange|common stock|nasdaq|new york stock exchange|nyse", t, re.I) or re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t.strip()):
                        out.append(s)
        return out
    except Exception:
        return []
