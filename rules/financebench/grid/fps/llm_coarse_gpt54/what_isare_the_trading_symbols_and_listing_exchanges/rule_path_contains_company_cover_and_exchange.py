def rule_path_contains_company_cover_and_exchange(doc: dict) -> list[dict]:
    """Match spans on page 1 under company cover-page paths that mention exchange or symbol cues."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            path = ((s.get("structure") or {}).get("path_text") or "")
            txt = (s.get("text") or "")
            if s.get("page_no") == 1 and path:
                if re.search(r"exchange|Trading Symbol|Trading symbol|NASDAQ|NYSE|New York Stock Exchange", txt + " " + path, re.I):
                    out.append(s)
        return out
    except Exception:
        return []
