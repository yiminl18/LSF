def rule_pages_near_item8_from_toc_and_net_income(doc: dict) -> list[dict]:
    """Match spans near Item 8 pages that also mention net income/earnings/loss."""
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
                if re.search(r"item\s*8", joined, re.I):
                    for v in cols.values():
                        if re.fullmatch(r"\d{1,3}", v.strip()):
                            p = int(v.strip())
                            pages.update({p, p + 1, p + 2, p + 3})
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no") in pages
            and re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
