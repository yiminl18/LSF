def rule_near_balance_sheet_pages_from_toc(doc: dict) -> list[dict]:
    """Match tables on or near the page listed for Consolidated Balance Sheet in the table of contents."""
    import re
    try:
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            m = re.search(r"Consolidated Balance Sheet[s]?\s*\|\s*(\d+)\s*\|", txt, re.I)
            if m:
                pages.add(int(m.group(1)))
        out = []
        if not pages:
            return out
        near = {p for x in pages for p in (x - 1, x, x + 1, x + 2)}
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") in near:
                out.append(span)
        return out
    except Exception:
        return []
