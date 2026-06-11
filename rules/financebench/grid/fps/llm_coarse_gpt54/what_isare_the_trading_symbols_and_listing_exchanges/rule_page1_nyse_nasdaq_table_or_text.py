def rule_page1_nyse_nasdaq_table_or_text(doc: dict) -> list[dict]:
    """Match page-1 text or table spans that directly mention NASDAQ/NYSE in the securities registration area."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("page_no") != 1:
                continue
            txt = (s.get("text") or "")
            if s.get("label") == "table":
                cells = (((s.get("table_data") or {}).get("cells")) or [])
                txt += " " + " ".join((c.get("text") or "") for c in cells)
            if re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq Global Select Market|Nasdaq Stock Market LLC", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
