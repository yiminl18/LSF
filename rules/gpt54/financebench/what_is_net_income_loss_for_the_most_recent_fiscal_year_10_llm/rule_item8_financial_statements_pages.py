def rule_item8_financial_statements_pages(doc: dict) -> list[dict]:
    """Match spans on pages associated with Item 8 / Financial Statements from the table of contents."""
    import re
    try:
        item8_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), {})[c.get("col")] = c.get("text", "") or ""
            for r, cols in rows.items():
                joined = " | ".join(cols.get(i, "") for i in sorted(cols))
                if re.search(r"item\s*8", joined, re.I) and re.search(r"financial statements", joined, re.I):
                    for v in cols.values():
                        if re.fullmatch(r"\d{1,3}", v.strip()):
                            item8_pages.add(int(v.strip()))
        if not item8_pages:
            return []
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in item8_pages or span.get("page_no") in {p + 1 for p in item8_pages}:
                out.append(span)
        return out
    except Exception:
        return []
