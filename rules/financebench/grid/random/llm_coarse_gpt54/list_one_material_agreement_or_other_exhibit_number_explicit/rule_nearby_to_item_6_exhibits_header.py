def rule_nearby_to_item_6_exhibits_header(doc: dict) -> list[dict]:
    """Match spans within a short window after a 10-Q Item 6 Exhibits header."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = span.get("text") or ""
            if re.search(r"item\s*6", txt, re.I) and re.search(r"\bexhibits?\b", txt, re.I):
                for j in range(i, min(i + 10, len(texts))):
                    out.append(texts[j])
    except Exception:
        return []
    return out
