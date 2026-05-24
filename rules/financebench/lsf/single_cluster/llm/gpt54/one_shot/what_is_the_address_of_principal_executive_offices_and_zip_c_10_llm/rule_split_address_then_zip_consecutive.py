def rule_split_address_then_zip_consecutive(doc: dict) -> list[dict]:
    """Match consecutive page-1 spans where one looks like address and the next is ZIP or vice versa."""
    try:
        import re
        spans = doc.get("texts", [])
        out_idx = set()
        for i in range(len(spans) - 1):
            a, b = spans[i], spans[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta = (a.get("text") or "").strip()
            tb = (b.get("text") or "").strip()
            la, lb = ta.lower(), tb.lower()
            a_addr = re.search(r"\b\d{1,6}\b", ta) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", la)
            b_zip = re.fullmatch(r"\d{5}(?:-\d{4})?", tb) or "(zip code)" in lb or re.match(r"\d{5}(?:-\d{4})?\s+\(zip code\)", tb.lower())
            b_addr = re.search(r"\b\d{1,6}\b", tb) and re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", lb)
            a_zip = re.fullmatch(r"\d{5}(?:-\d{4})?", ta) or "(zip code)" in la
            if a_addr and b_zip:
                out_idx.update([i, i + 1])
            if a_zip and b_addr:
                out_idx.update([i, i + 1])
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
