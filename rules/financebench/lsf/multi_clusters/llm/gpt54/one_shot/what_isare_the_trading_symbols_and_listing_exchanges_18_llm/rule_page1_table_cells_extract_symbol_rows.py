def rule_page1_table_cells_extract_symbol_rows(doc: dict) -> list[dict]:
    """Match table spans with rows containing common stock/ordinary shares and symbol/exchange values."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append((c.get("text") or "").lower())
            for vals in rows.values():
                joined = " | ".join(vals)
                if (
                    ("common stock" in joined or "ordinary shares" in joined or "notes due" in joined)
                    and ("nasdaq" in joined or "stock exchange" in joined)
                ):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
