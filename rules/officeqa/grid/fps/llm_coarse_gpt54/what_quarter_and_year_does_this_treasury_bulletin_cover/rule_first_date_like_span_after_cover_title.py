def rule_first_date_like_span_after_cover_title(doc: dict) -> list[dict]:
    """Match the first date-like span after the first Treasury Bulletin title occurrence."""
    import re
    try:
        texts = doc.get("texts", [])
        title_idx = None
        for i, s in enumerate(texts):
            if re.search(r"treasury\s+bulletin", (s.get("text") or "").strip(), re.I):
                title_idx = i
                break
        if title_idx is None:
            return []
        pat = re.compile(
            r"\b("
            r"January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Spring|Summer|Fall|Winter|"
            r"First|Second|Third|Fourth|1st|2nd|3rd|4th"
            r")\b",
            re.I,
        )
        out = []
        for j in range(title_idx + 1, min(title_idx + 12, len(texts))):
            txt = (texts[j].get("text") or "").strip()
            if txt and pat.search(txt):
                out.append(texts[j])
                break
        return out
    except Exception:
        return []
