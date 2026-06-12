def rule_page_with_ffo1_and_in_millions(doc: dict) -> list[dict]:
    """Match spans on pages where a nearby header says FFO-1 and nearby text says in millions."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            txt = span.get("text") or ""
            if re.search(r'(Table\s*)?FFO[-\s]?1', txt, re.I) or re.search(r'Summary of Fiscal Operations', txt, re.I):
                pages.add(span.get("page_no"))
        for span in texts:
            if span.get("page_no") in pages:
                txt = span.get("text") or ""
                if span.get("label") == "table" and re.search(r'(receipts|outlays|fiscal year to date|actual fiscal year to date)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
