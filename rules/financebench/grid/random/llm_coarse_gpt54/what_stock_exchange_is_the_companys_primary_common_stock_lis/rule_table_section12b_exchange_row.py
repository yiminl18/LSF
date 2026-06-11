def rule_table_section12b_exchange_row(doc: dict) -> list[dict]:
    """Return table cells from page-1 Section 12(b) tables whose row contains a common stock class and exchange value."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        exchange_pats = [
            r"\bnew york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
        ]
        for span in texts:
            if span.get("label") != "table":
                continue
            if span.get("page_no") != 1:
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for r, row_cells in rows.items():
                row_text = " ".join((c.get("text") or "") for c in row_cells).lower()
                if "common stock" in row_text or "ordinary shares" in row_text:
                    for c in row_cells:
                        low = (c.get("text") or "").lower()
                        if any(re.search(p, low) for p in exchange_pats):
                            out.append({
                                "text": c.get("text"),
                                "label": "table_cell",
                                "page_no": span.get("page_no"),
                                "source_span": span,
                                "cell": c,
                                "structure": span.get("structure", {}),
                            })
        return out
    except Exception:
        return []
