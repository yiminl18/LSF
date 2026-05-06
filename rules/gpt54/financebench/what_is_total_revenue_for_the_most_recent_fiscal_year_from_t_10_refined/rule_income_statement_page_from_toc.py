def rule_income_statement_page_from_toc(doc: dict) -> list[dict]:
    """Match spans on the page named in the TOC for Consolidated Statement of Income / Operations / Earnings."""
    try:
        texts = doc.get("texts", [])
        target_pages = set()
        for span in texts:
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            if "table of contents" not in txt and "index" not in txt and "table of contents" not in path and "index" not in path:
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_join = " | ".join((c.get("text") or "") for c in row_cells).lower()
                if any(k in row_join for k in [
                    "consolidated statement of income",
                    "consolidated statements of income",
                    "consolidated statement of operations",
                    "consolidated statements of operations",
                    "consolidated statement of earnings",
                    "consolidated statements of earnings"
                ]):
                    for c in row_cells:
                        t = (c.get("text") or "").strip()
                        if t.isdigit():
                            target_pages.add(int(t))
        out = []
        for span in texts:
            if span.get("page_no") in target_pages:
                out.append(span)
        return out
    except Exception:
        return []
