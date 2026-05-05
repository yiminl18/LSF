def rule_exhibit_index_neighbor_window(doc: dict) -> list[dict]:
    """Return spans within a small window around any Exhibit Index header."""
    import re
    try:
        texts = doc.get("texts", [])
        idxs = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if re.search(r"\bexhibit index\b", txt, re.I) or "exhibit index" in path.lower():
                idxs.append(i)
        if not idxs:
            return []
        out = []
        seen = set()
        for i in idxs:
            for j in range(max(0, i - 3), min(len(texts), i + 8)):
                if j not in seen:
                    out.append(texts[j])
                    seen.add(j)
        return out
    except Exception:
        return []
