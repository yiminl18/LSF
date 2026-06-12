def rule_numeric_reading_after_michigan_mention(doc: dict) -> list[dict]:
    """Match numeric spans within a short window after a Michigan/Reuters/sentiment mention."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        idxs = set()
        for i, span in enumerate(texts):
            text = (span.get("text") or "")
            if re.search(r"(university of michigan|michigan/reuters|consumer sentiment|reuters consumer sentiment)", text, re.I):
                for j in range(i, min(len(texts), i + 4)):
                    t2 = (texts[j].get("text") or "")
                    if re.search(r"\b\d{2,3}\.?\d*\b", t2):
                        idxs.add(j)
        for j in sorted(idxs):
            out.append(texts[j])
        return out
    except Exception:
        return []
