def rule_nearby_to_exhibit_index_header(doc: dict) -> list[dict]:
    """Match spans within a short window after an Exhibit Index header."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = span.get("text") or ""
            if re.search(r"\bexhibit index\b", txt, re.I):
                for j in range(i, min(i + 8, len(texts))):
                    out.append(texts[j])
    except Exception:
        return []
    return out
