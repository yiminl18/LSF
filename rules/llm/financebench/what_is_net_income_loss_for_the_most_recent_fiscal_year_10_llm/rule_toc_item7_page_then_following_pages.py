def rule_toc_item7_page_then_following_pages(doc: dict) -> list[dict]:
    """Match spans on Item 7 pages and nearby pages, where some filers state net income in MD&A."""
    import re
    try:
        pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), {})[c.get("col")] = c.get("text", "") or ""
            for cols in rows.values():
                joined = " | ".join(cols.get(i, "") for i in sorted(cols))
                if re.search(r"item\s*7", joined, re.I) and re.search(r"management'?s discussion", joined, re.I):
                    for v in cols.values():
                        if re.fullmatch(r"\d{1,3}", v.strip()):
                            p = int(v.strip())
                            pages.update({p, p + 1, p + 2})
        if not pages:
            return []
        return [s for s in doc.get("texts", []) if s.get("page_no") in pages]
    except Exception:
        return []
