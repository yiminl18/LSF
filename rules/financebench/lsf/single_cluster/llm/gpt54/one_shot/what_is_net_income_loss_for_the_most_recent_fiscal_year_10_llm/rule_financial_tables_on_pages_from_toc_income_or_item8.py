def rule_financial_tables_on_pages_from_toc_income_or_item8(doc: dict) -> list[dict]:
    """Match tables on pages identified by TOC as income statement or Item 8 pages."""
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
                if re.search(r"(item\s*8.*financial statements)|(statement of income)|(statement of operations)", joined, re.I):
                    for v in cols.values():
                        if re.fullmatch(r"\d{1,3}", v.strip()):
                            pages.add(int(v.strip()))
        return [s for s in doc.get("texts", []) if s.get("label") == "table" and s.get("page_no") in pages]
    except Exception:
        return []
