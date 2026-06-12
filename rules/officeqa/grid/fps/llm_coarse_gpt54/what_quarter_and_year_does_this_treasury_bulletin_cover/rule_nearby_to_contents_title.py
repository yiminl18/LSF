def rule_nearby_to_contents_title(doc: dict) -> list[dict]:
    """Match date-like spans within a short window after a Contents/Table of Contents header."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        pat = re.compile(
            r"\b("
            r"March|June|September|December|January|February|April|May|July|August|October|November|"
            r"Spring|Summer|Fall|Winter|"
            r"First Quarter, Fiscal \d{4}|Second Quarter, Fiscal \d{4}|Third Quarter, Fiscal \d{4}|Fourth Quarter, Fiscal \d{4}"
            r")\b",
            re.I,
        )
        for i, s in enumerate(texts):
            txt = (s.get("text") or "").strip().lower()
            if txt in {"contents", "table of contents"}:
                for j in range(i + 1, min(i + 8, len(texts))):
                    cand = texts[j]
                    ctext = (cand.get("text") or "").strip()
                    if pat.search(ctext):
                        out.append(cand)
        return out
    except Exception:
        return []
