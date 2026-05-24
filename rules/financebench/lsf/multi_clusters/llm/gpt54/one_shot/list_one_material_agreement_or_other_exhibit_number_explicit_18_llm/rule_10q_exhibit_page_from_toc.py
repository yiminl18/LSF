def rule_10q_exhibit_page_from_toc(doc: dict) -> list[dict]:
    """Use 10-Q TOC Item 6 Exhibits page number to return spans on or near that page."""
    import re
    try:
        texts = doc.get("texts", [])
        target_pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " ".join((c.get("text") or "") for c in row_cells)
                if re.search(r"\bitem\s*6\b", row_text, re.I) and re.search(r"\bexhibits?\b", row_text, re.I):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if re.fullmatch(r"\d{1,4}", t):
                            p = int(t)
                            target_pages.update({p, p + 1})
        if not target_pages:
            return []
        return [s for s in texts if s.get("page_no") in target_pages]
    except Exception:
        return []
