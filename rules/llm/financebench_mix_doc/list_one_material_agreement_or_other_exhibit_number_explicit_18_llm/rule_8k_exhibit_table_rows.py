def rule_8k_exhibit_table_rows(doc: dict) -> list[dict]:
    """Match 8-K exhibit tables listing Exhibit No. and Description."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            txt = " ".join((c.get("text") or "") for c in cells)
            if re.search(r"\bexhibit no\.?\b", txt, re.I) and re.search(r"\bdescription\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
