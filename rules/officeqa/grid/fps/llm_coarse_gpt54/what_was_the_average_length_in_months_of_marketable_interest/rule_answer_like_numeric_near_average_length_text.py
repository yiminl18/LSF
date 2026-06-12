def rule_answer_like_numeric_near_average_length_text(doc: dict) -> list[dict]:
    """Return text spans with short numeric content near a nearby span mentioning average length."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if not re.fullmatch(r'\d{1,3}', txt):
                continue
            window = texts[max(0, i - 3): min(len(texts), i + 4)]
            joined = " ".join((s.get("text") or "") for s in window).lower()
            if "average length" in joined and "marketable" in joined:
                out.append(span)
    except Exception:
        return []
    return out
