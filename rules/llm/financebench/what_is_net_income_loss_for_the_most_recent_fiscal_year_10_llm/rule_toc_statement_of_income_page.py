def rule_toc_statement_of_income_page(doc: dict) -> list[dict]:
    """Match spans on the page listed in the TOC for Consolidated Statement of Income/Operations."""
    import re
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), {})[c.get("col")] = c.get("text", "") or ""
            for cols in rows.values():
                joined = " | ".join(cols.get(i, "") for i in sorted(cols))
                if re.search(r"(consolidated )?(statement|statements) of (income|operations|earnings)", joined, re.I):
                    for v in cols.values():
                        if re.fullmatch(r"\d{1,3}", v.strip()):
                            target_pages.add(int(v.strip()))
        if not target_pages:
            return []
        return [s for s in doc.get("texts", []) if s.get("page_no") in target_pages]
    except Exception:
        return []
