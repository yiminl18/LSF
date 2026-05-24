def rule_page1_table_cells_return_matching_rows(doc: dict) -> list[dict]:
    """Match table spans where rows contain both a symbol-like cell and an exchange cell."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        sym = re.compile(r"^[A-Z]{1,6}(?:/[0-9]{2})?$")
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                vals = [(c.get("text") or "").strip() for c in row_cells]
                has_sym = any(sym.match(v) for v in vals)
                has_ex = any(("exchange" in v.lower() or "nasdaq" in v.lower()) for v in vals)
                if has_sym and has_ex:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
