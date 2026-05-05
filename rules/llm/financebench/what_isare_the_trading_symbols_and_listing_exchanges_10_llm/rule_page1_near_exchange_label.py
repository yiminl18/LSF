def rule_page1_near_exchange_label(doc: dict) -> list[dict]:
    """Match spans within a small window around a page 1 exchange label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"name of each exchange|exchange on which registered", txt, re.I):
                for j in range(max(0, i - 4), min(len(texts), i + 6)):
                    s = texts[j]
                    t = (s.get("text") or "")
                    if re.search(r"exchange|registered|nasdaq|new york stock exchange|nyse", t, re.I) or re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t.strip()):
                        out.append(s)
        return out
    except Exception:
        return []
