def rule_table_exchange_value_cells(doc: dict) -> list[dict]:
    """Return synthetic span-like dicts for table cells under exchange-registration columns that contain exchange names."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        exchange_pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
            r"\bnasdaq global market\b",
            r"\bthe nasdaq global market\b",
        ]
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            header_cols = set()
            for c in cells:
                txt = (c.get("text") or "").lower()
                if "name of each exchange on which registered" in txt:
                    header_cols.add(c.get("col"))
            if not header_cols:
                continue
            for c in cells:
                if c.get("col") in header_cols:
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
