def rule_table_row_header_title_of_each_class(doc: dict) -> list[dict]:
    """Match tables whose cells include 'Title of each class' row/column header."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                txt = (c.get("text") or "").lower()
                if "title of each class" in txt or "title of each class" == txt:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
