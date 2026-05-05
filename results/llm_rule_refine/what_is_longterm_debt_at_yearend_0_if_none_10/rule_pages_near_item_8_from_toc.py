def rule_pages_near_item_8_from_toc(doc: dict) -> list[dict]:
    """Match tables on or just after the page where Item 8 / Financial Statements begins, inferred from TOC text."""
    import re
    try:
        texts = doc.get("texts", [])
        item8_pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            m = re.search(r"Item 8\..{0,120}?Financial Statements.*?\|\s*(\d+)\s*\|", txt, re.I | re.S)
            if m:
                item8_pages.add(int(m.group(1)))
        out = []
        if not item8_pages:
            return out
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") in {p for x in item8_pages for p in (x, x + 1, x + 2)}:
                out.append(span)
        return out
    except Exception:
        return []
