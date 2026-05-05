def rule_page1_address_near_zip(doc: dict) -> list[dict]:
    """Return spans near a page-1 ZIP-only span or ZIP label."""
    try:
        import re
        spans = doc.get("texts", [])
        out_idx = set()
        for i, s in enumerate(spans):
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).strip()
            low = txt.lower()
            if re.fullmatch(r"\d{5}(?:-\d{4})?", txt) or "(zip code)" in low or re.search(r"\b\d{5}(?:-\d{4})?\s+\(zip code\)", low):
                for j in range(max(0, i - 3), min(len(spans), i + 3)):
                    if spans[j].get("page_no") == 1:
                        out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
