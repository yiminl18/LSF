def rule_page1_table_cells_exchange_names(doc: dict) -> list[dict]:
    """Return synthetic span-like dicts for any page-1 table cell containing a known exchange name."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        pats = [
            r"\bnew york stock exchange\b",
            r"\bthe new york stock exchange\b",
            r"\bnasdaq\b",
            r"\bnasdaq global select market\b",
            r"\bthe nasdaq global select market\b",
        ]
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") == 1:
                for c in (((span.get("table_data") or {}).get("cells")) or []):
                    low = (c.get("text") or "").lower()
                    if any(re.search(p, low) for p in pats):
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
