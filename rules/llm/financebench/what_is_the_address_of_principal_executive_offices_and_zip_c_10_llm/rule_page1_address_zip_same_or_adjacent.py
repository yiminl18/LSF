def rule_page1_address_zip_same_or_adjacent(doc: dict) -> list[dict]:
    """Match spans where address and ZIP appear in the same span or in adjacent spans."""
    try:
        import re
        spans = doc.get("texts", [])
        out_idx = set()
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low):
                if re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                    out_idx.add(i)
                if i + 1 < len(spans) and spans[i + 1].get("page_no") == 1:
                    nxt = (spans[i + 1].get("text") or "").strip()
                    if re.fullmatch(r"\d{5}(?:-\d{4})?", nxt) or "zip code" in nxt.lower():
                        out_idx.add(i)
                        out_idx.add(i + 1)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
