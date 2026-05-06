def rule_table_12b_headers(doc: dict) -> list[dict]:
    """Match tables whose cells contain trading symbol / exchange registration headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            joined = " ".join((c.get("text") or "") for c in cells)
            if re.search(r"trading symbol|name of each exchange|exchange on which registered|title of each class", joined, re.I):
                out.append(span)
        return out
    except Exception:
        return []
