def rule_page1_cover_tables_with_symbol_or_exchange(doc: dict) -> list[dict]:
    """Match page-1 tables whose cells mention trading symbols or exchanges."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table" or s.get("page_no") != 1:
                continue
            cells = (((s.get("table_data") or {}).get("cells")) or [])
            joined = " | ".join((c.get("text") or "") for c in cells)
            if re.search(r"Trading Symbol|Trading symbol|exchange on which registered|NASDAQ|NYSE|New York Stock Exchange", joined, re.I):
                out.append(s)
        return out
    except Exception:
        return []
