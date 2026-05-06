def rule_ein_and_state_adjacent_spans(doc: dict) -> list[dict]:
    """Match adjacent page-1 spans where one looks like an EIN and the other is a nearby state/jurisdiction label or value."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 1):
            a = texts[i]
            b = texts[i + 1]
            ta = (a.get("text", "") or "").strip()
            tb = (b.get("text", "") or "").strip()
            if a.get("page_no") == 1 and b.get("page_no") == 1:
                if re.fullmatch(r"\d{2}-\d{7}", ta) and (
                    re.search(r"i\.?r\.?s\.?|employer identification|state or other jurisdiction", tb, re.I)
                    or re.fullmatch(r"[A-Za-z][A-Za-z .,&()\-]{1,40}", tb)
                ):
                    out.extend([a, b])
        return out
    except Exception:
        return []
