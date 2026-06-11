def rule_page1_exchange_after_section12b_table_header(doc: dict) -> list[dict]:
    """Return page-1 table cells in columns headed by exchange-registration wording."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table" or span.get("page_no") != 1:
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            exchange_cols = []
            for c in cells:
                if "name of each exchange on which registered" in (c.get("text") or "").lower():
                    exchange_cols.append(c.get("col"))
            for c in cells:
                if c.get("col") in exchange_cols and not c.get("is_column_header"):
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
