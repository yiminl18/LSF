def rule_exhibit_item_neighbor_window(doc: dict) -> list[dict]:
    """Return spans within a small window around Item 15/Item 6 exhibit headers."""
    import re
    try:
        texts = doc.get("texts", [])
        idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if re.search(r"\bitem\s*(15|6)\b", txt, re.I) and re.search(r"\bexhibit", txt, re.I):
                idxs.append(i)
        if not idxs:
            return []
        out = []
        seen = set()
        for i in idxs:
            for j in range(max(0, i - 2), min(len(texts), i + 10)):
                if j not in seen:
                    out.append(texts[j])
                    seen.add(j)
        return out
    except Exception:
        return []
