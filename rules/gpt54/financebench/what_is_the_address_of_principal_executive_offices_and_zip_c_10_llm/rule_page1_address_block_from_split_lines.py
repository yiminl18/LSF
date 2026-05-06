def rule_page1_address_block_from_split_lines(doc: dict) -> list[dict]:
    """Match split-line address blocks where street, city/state, and ZIP are separate nearby spans."""
    try:
        import re
        spans = doc.get("texts", [])
        out_idx = set()
        for i, s in enumerate(spans):
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low):
                for j in range(i, min(len(spans), i + 4)):
                    if spans[j].get("page_no") == 1:
                        out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
