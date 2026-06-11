def rule_nearby_to_item_9_01_header(doc: dict) -> list[dict]:
    """Match spans within a short window after an 8-K Item 9.01 header."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = span.get("text") or ""
            if re.search(r"item\s*9\.?01", txt, re.I):
                for j in range(i, min(i + 12, len(texts))):
                    out.append(texts[j])
    except Exception:
        return []
    return out
