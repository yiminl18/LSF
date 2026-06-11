def rule_page1_table_12b(doc: dict) -> list[dict]:
    """Match page-1 tables that likely contain the trading symbol and exchange answer."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table" or span.get("page_no") != 1:
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            txt = " | ".join((c.get("text", "") or "") for c in cells)
            if re.search(r"trading symbol", txt, re.I) or re.search(r"name of each exchange on which registered", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
