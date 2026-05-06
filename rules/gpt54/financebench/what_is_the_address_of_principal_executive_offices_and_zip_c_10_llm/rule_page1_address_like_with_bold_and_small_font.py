def rule_page1_address_like_with_bold_and_small_font(doc: dict) -> list[dict]:
    """Match small-font bold cover-page address spans, a common SEC template pattern."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            size = float(span.get("size") or 0)
            if not (6.0 <= size <= 9.5):
                continue
            if span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{1,6}\b", txt) and (
                re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low)
                or re.search(r"\b\d{5}(?:-\d{4})?\b", txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
