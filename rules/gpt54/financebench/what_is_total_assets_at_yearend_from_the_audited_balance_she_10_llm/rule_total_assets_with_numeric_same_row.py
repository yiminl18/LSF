def rule_total_assets_with_numeric_same_row(doc: dict) -> list[dict]:
    """Match tables where a total assets row also contains numeric values in the same row."""
    import re
    try:
        out = []
        num_re = re.compile(r"^\$?\(?\d[\d,.\s]*\)?$")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                row_text = " ".join((c.get("text") or "").lower() for c in row_cells)
                if "total assets" not in row_text:
                    continue
                numeric_count = 0
                for c in row_cells:
                    txt = (c.get("text") or "").strip()
                    if num_re.match(txt.replace(" ", "")) and any(ch.isdigit() for ch in txt):
                        numeric_count += 1
                if numeric_count >= 1:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
