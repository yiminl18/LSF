def rule_exhibit_answer_pages_from_toc(doc: dict) -> list[dict]:
    """Use TOC exhibit page references to collect spans on the target exhibit pages."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        target_pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " ".join((c.get("text") or "") for c in row_cells)
                if re.search(r"item\s*15", row_text, re.I) and re.search(r"exhibit", row_text, re.I):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if re.fullmatch(r"\d{1,4}", t):
                            target_pages.add(int(t))
                if re.search(r"\bexhibit index\b", row_text, re.I):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if re.fullmatch(r"\d{1,4}", t):
                            target_pages.add(int(t))
        for span in texts:
            if (span.get("page_no") or 0) in target_pages:
                out.append(span)
    except Exception:
        return []
    return out
