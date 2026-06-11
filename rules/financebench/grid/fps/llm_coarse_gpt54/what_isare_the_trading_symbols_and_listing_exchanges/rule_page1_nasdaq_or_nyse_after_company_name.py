def rule_page1_nasdaq_or_nyse_after_company_name(doc: dict) -> list[dict]:
    """Match page-1 exchange mentions appearing under the main company cover section."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            path = ((s.get("structure") or {}).get("path_text") or "")
            txt = (s.get("text") or "")
            if s.get("page_no") == 1 and path and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", txt, re.I):
                out.append(s)
        return out
    except Exception:
        return []
