def rule_state_and_ein_adjacent_spans(doc: dict) -> list[dict]:
    """Match adjacent page-1 spans where one looks like a state and the next like an EIN or label."""
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
                if re.fullmatch(r"[A-Za-z][A-Za-z .,&()\-]{1,40}", ta) and (
                    re.fullmatch(r"\d{2}-\d{7}", tb) or re.search(r"state or other jurisdiction|i\.?r\.?s\.?|employer identification", tb, re.I)
                ):
                    out.extend([a, b])
        return out
    except Exception:
        return []
