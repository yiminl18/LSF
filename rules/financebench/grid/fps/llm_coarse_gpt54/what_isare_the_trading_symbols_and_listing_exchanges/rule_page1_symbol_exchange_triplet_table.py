def rule_page1_symbol_exchange_triplet_table(doc: dict) -> list[dict]:
    """Match 3-column page-1 tables with class, symbol, and exchange columns."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table" or s.get("page_no") != 1:
                continue
            td = s.get("table_data") or {}
            if td.get("num_cols") == 3:
                cells = td.get("cells") or []
                joined = " | ".join((c.get("text") or "") for c in cells)
                if re.search(r"Title of each class", joined, re.I) and re.search(r"Trading Symbol", joined, re.I) and re.search(r"exchange", joined, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
