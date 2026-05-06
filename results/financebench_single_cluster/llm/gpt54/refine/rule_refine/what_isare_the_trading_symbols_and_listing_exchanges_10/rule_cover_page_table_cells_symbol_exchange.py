def rule_cover_page_table_cells_symbol_exchange(doc: dict) -> list[dict]:
    """Match table spans whose cells contain trading symbol / exchange registration headers."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            cell_text = " ".join((c.get("text") or "") for c in cells).lower()
            if any(k in cell_text for k in [
                "trading symbol",
                "trading symbol(s)",
                "name of each exchange on which registered",
                "name of exchange on which registered",
                "name of each exchange",
            ]):
                out.append(span)
        return out
    except Exception:
        return []
