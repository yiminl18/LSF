def rule_table_with_12b_headers(doc: dict) -> list[dict]:
    """Match tables that look like the Section 12(b) securities table with class/symbol/exchange headers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            texts = " | ".join((c.get("text", "") or "") for c in cells)
            if re.search(r"title of each class", texts, re.I) and re.search(r"trading symbol", texts, re.I) and re.search(r"exchange", texts, re.I):
                out.append(span)
        return out
    except Exception:
        return []
