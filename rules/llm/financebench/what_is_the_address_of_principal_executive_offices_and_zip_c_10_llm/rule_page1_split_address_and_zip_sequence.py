def rule_page1_split_address_and_zip_sequence(doc: dict) -> list[dict]:
    """Match page-1 sequences where address and ZIP are split across adjacent spans."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for i in range(len(spans) - 1):
            a = spans[i]
            b = spans[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta = (a.get("text") or "").strip()
            tb = (b.get("text") or "").strip()
            la = ta.lower()
            lb = tb.lower()
            if re.search(r"\b\d{1,6}\b", ta) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", la):
                if re.fullmatch(r"\d{5}(?:-\d{4})?", tb) or "(zip code)" in lb or "zip code" in lb:
                    out.extend([a, b])
            if "(address of principal executive offices)" in ((a.get("text_span") or "").lower()):
                out.append(a)
        return out
    except Exception:
        return []
